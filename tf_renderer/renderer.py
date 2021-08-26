import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow_graphics.rendering.reflectance import lambertian, phong
from tensorflow_graphics.geometry.representation import triangle, grid

def point_in_triangle(points, triangles):
    """
    points: [..., 2]
    triangles: [..., 3, 2]
    """

    def sign(p1, p2, p3):
        e1 = p1 - p3
        e2 = p2 - p3
        return e1[...,0] * e2[...,1] - e2[...,0] * e1[...,1]

    b1 = sign(points, triangles[...,0,:], triangles[...,1,:]) <= 0.
    b2 = sign(points, triangles[...,1,:], triangles[...,2,:]) <= 0.
    b3 = sign(points, triangles[...,2,:], triangles[...,0,:]) <= 0.
    return tf.logical_and(tf.equal(b1, b2), tf.equal(b2, b3))

def assign_triangles_to_pixels_bb(triangles, triangle_depths, image_size):
    # generate grid in bounding box
    starts = tf.minimum(tf.maximum(tf.cast(tf.math.floor(tf.reduce_min(triangles[...,::-1], axis=[0,1])), tf.int32), 0), image_size)
    stops = tf.minimum(tf.maximum(tf.cast(tf.math.ceil(tf.reduce_max(triangles[...,::-1], axis=[0,1])), tf.int32), 0), image_size)
    grid = tf.cast(tf.stack(tf.meshgrid(tf.range(starts[1], stops[1]), tf.range(starts[0], stops[0])), axis=-1), tf.float32)

    # find face indices projected onto the bounding box
    in_triangle = point_in_triangle(grid[tf.newaxis], triangles[:,tf.newaxis,tf.newaxis])
    triangle_depth = tf.where(in_triangle, triangle_depths[:,tf.newaxis,tf.newaxis], float('inf'))
    triangle_depth = tf.where(triangle_depth > 0., triangle_depth, float('inf'))
    nearest_inds = tf.argmin(triangle_depth)
    nearest_depths = tf.gather(triangle_depths, nearest_inds)
    valid = tf.reduce_any(in_triangle, axis=0)
    nearest_depths = tf.where(valid, nearest_depths, float('inf'))

    # pad with invalid values
    paddings = [[starts[0], image_size[0]-stops[0]], [starts[1], image_size[1]-stops[1]]]
    nearest_inds = tf.pad(nearest_inds, paddings, constant_values=0)
    nearest_depths = tf.pad(nearest_depths, paddings, constant_values=float('inf'))
    valid = tf.pad(valid, paddings, constant_values=False)
    return nearest_inds, nearest_depths, valid

def assign_triangles_to_pixels(vertices2d, depths, image_size, faces, face_inds=None):
    depth_buf = tf.ones(image_size) * float('inf')
    inds = tf.zeros(image_size, dtype=tf.int32)
    valid = tf.zeros(image_size, dtype=tf.bool)

    if face_inds is None:
        face_inds = [tf.range(tf.shape(faces)[0])]

    for _face_inds in face_inds:
        part_faces = tf.gather(faces, _face_inds)

        # gather triangles of part
        triangles = tf.gather(vertices2d, part_faces)
        triangle_depths = tf.reduce_mean(tf.gather(depths, part_faces), axis=1)
        # assign to pixels
        _inds, _depth_buf, _valid = assign_triangles_to_pixels_bb(triangles, triangle_depths, image_size)
        _inds = tf.gather(_face_inds, _inds)
        # merge results
        inds = tf.where(depth_buf < _depth_buf, inds, _inds)
        depth_buf = tf.minimum(depth_buf, _depth_buf)
        valid = tf.logical_or(valid, _valid)
    return inds, depth_buf, valid

def ray_triangle_intersection(ray_origins, ray_vectors, triangles, return_coords=False):
    """ Calculate where the ray intersects with the triangle.
    ray_origins: [...,3]
    ray_vectors: [...,3]
    triangles: [...,3,3]
    return_coords: bool
        Whether return pyramidal coordinates (t,u,v) or not.
    """
    vertex0 = triangles[...,0,:]
    vertex1 = triangles[...,1,:]
    vertex2 = triangles[...,2,:]
    edge1 = vertex1 - vertex0
    edge2 = vertex2 - vertex0
    s = ray_origins - vertex0

    h = tf.linalg.cross(ray_vectors, edge2)
    a = tf.reduce_sum(edge1 * h, axis=-1, keepdims=True)
    is_parallel = tf.abs(a) < 1e-7
    q = tf.linalg.cross(s, edge1)
    t = tf.reduce_sum(edge2 * q, axis=-1, keepdims=True) / a
    p = ray_origins + ray_vectors * t
    p = tf.where(is_parallel, (vertex0+vertex1+vertex2)/3., p)
    if return_coords:
        u = tf.reduce_sum(s * h, axis=-1, keepdims=True) / a
        v = tf.reduce_sum(ray_vectors * q, axis=-1, keepdims=True) / a
        u = tf.where(is_parallel, 0.5, u)
        v = tf.where(is_parallel, 0.5, v)
        coords = tf.concat([t, u, v], axis=-1)
        return p, coords
    else:
        return p

def reduce_faces(faces, face_inds, valid=None):
    if valid is None:
        old_face_inds = tf.reshape(face_inds, [-1])
    else:
        old_face_inds = tf.boolean_mask(face_inds, valid)
    new_face_inds, _ = tf.unique(tf.reshape(old_face_inds, [-1]))
    new_faces = tf.gather(faces, new_face_inds)
    n_new_faces = tf.shape(new_faces)[0]
    all2new = tf.scatter_nd(new_face_inds[:,tf.newaxis], tf.range(n_new_faces), tf.shape(faces)[0:1])
    new_face_inds = tf.gather(all2new, face_inds)

    return new_faces, new_face_inds

def vertex_normals(vertices, faces):
    face_normals = triangle.normal(
        tf.gather(vertices, faces[:,0]),
        tf.gather(vertices, faces[:,1]),
        tf.gather(vertices, faces[:,2]), clockwise=True, normalize=True)
    n_verts = tf.shape(vertices)[0]
    n_faces = tf.shape(faces)[0]
    indices = tf.tile(tf.range(n_faces)[:,tf.newaxis], [1,3])
    indices = tf.reshape(tf.stack([indices, faces], axis=-1), [-1,2])
    weights = tf.scatter_nd(indices, tf.ones(n_faces*3), [n_faces, n_verts])
    vert_normals = tf.matmul(weights, face_normals, transpose_a=True)
    vert_normals = tf.math.l2_normalize(vert_normals, axis=-1)
    return vert_normals

def tangent_vectors(vertices, texture_coords):
    a, c = tf.split(texture_coords[...,1,:] - texture_coords[...,0,:], 2, axis=-1)
    b, d = tf.split(texture_coords[...,2,:] - texture_coords[...,0,:], 2, axis=-1)
    #D = a * d - b * c
    
    edge1 = vertices[...,1,:] - vertices[...,0,:]
    edge2 = vertices[...,2,:] - vertices[...,0,:]

    tangent1 = edge1 * d + edge2 * (-c)
    tangent2 = edge1 * (-b) + edge2 * a
    tangent1 = tf.linalg.l2_normalize(tangent1, axis=-1)
    tangent2 = tf.linalg.l2_normalize(tangent2, axis=-1)
    return tangent1, tangent2

def interpolate_in_triangles(vectors, u, v):
    x0 = vectors[...,0,:]
    x1 = vectors[...,1,:]
    x2 = vectors[...,2,:]
    u = u[...,tf.newaxis]
    v = v[...,tf.newaxis]
    return x0 * (1.-u-v) + x1 * u + x2 * v

class Renderer(object):
    def __init__(self, camera, image_size):
        self.camera = camera
        self.image_size = tf.convert_to_tensor(image_size, dtype=tf.int32)

    def render(self, trimesh, light, ambient_light=0.2):
        all_verts = trimesh.vertices
        all_faces = trimesh.faces
        texture = trimesh.texture_map
        normal_map = trimesh.normal_map
        if texture is not None or normal_map is not None:
            all_tex_faces = trimesh.texture_faces
            tex_coords = trimesh.texture_coords
        else:
            albedo = trimesh.albedo
        n_verts = all_verts.shape[0]
        interpolate_albedo = texture is None and albedo.shape[0] == n_verts
        all_verts_proj, all_verts_depth = self.camera.project(all_verts)
        shininess = trimesh.shininess
        interpolate_shininess = shininess.shape[0] == n_verts
        vertex_properties = trimesh.vertex_properties

        inds, depth, valid = assign_triangles_to_pixels(
            all_verts_proj, all_verts_depth,
            self.image_size,
            all_faces, trimesh.face_groups)

        pixels = grid.generate([0,0], self.image_size-1, self.image_size)
        pixels = tf.cast(pixels, tf.float32)[:,:,::-1]
        camera_ray = self.camera.ray(pixels)
        outgoing_ray = tf.math.l2_normalize(-camera_ray, axis=-1)
        intersections, coords = ray_triangle_intersection(
            tf.zeros_like(camera_ray), camera_ray,
            tf.gather(all_verts, tf.gather(all_faces, inds)),
            return_coords = True)

        # filter out hidden faces
        visible_faces, visible_inds = reduce_faces(
            all_faces, inds, valid)
        visible_faces = tf.pad(visible_faces, [[0,1],[0,0]]) # add dummy to prevent error
        if texture is not None:
            visible_tex_faces, _ = reduce_faces(
                all_tex_faces, inds, valid)
            visible_tex_faces = tf.pad(visible_tex_faces, [[0,1],[0,0]]) # add dummy to prevent error

        vert_normals = vertex_normals(all_verts, visible_faces)

        # collect vertex properties to be interpolated
        vert_props = [vert_normals]
        prop_sizes = [3]
        if interpolate_albedo:
            vert_props.append(albedo)
            prop_sizes.append(3)
        if interpolate_shininess:
            vert_props.append(shininess)
            prop_sizes.append(1)
        if vertex_properties is not None:
            vert_props.append(vertex_properties)
            prop_sizes.append(tf.shape(vertex_properties)[-1])
        vert_props = tf.concat(vert_props, axis=-1)

        # interpolate in triangle coordinates
        vert_props = tf.gather(vert_props, tf.gather(visible_faces, visible_inds))
        u = coords[:,:,1]
        v = coords[:,:,2]
        pixel_props = interpolate_in_triangles(vert_props, u, v)

        # split vertex properties
        pixel_props = tf.split(pixel_props, prop_sizes, axis=-1)
        pixel_normals, *pixel_props = pixel_props
        if interpolate_albedo:
            albedo, *pixel_props = pixel_props
        if interpolate_shininess:
            shininess, *pixel_props = pixel_props
        if vertex_properties is not None:
            vertex_properties, *pixel_props = pixel_props
        pixel_normals = tf.math.l2_normalize(pixel_normals, axis=-1)

        # interpolate texture coordinates
        if texture is not None or normal_map is not None:
            tex_coords = tf.gather(
                tex_coords,
                tf.gather(visible_tex_faces, visible_inds))
            if normal_map is not None:
                tangent1, tangent2 = tangent_vectors(
                    tf.gather(all_verts, tf.gather(visible_faces, visible_inds)),
                    tex_coords)
            tex_coords = interpolate_in_triangles(tex_coords, u, v)

        # interpolate texture values
        if texture is not None:
            map_size = tf.cast(tf.shape(texture)[1::-1], tex_coords.dtype)
            albedo = tfa.image.interpolate_bilinear(
                texture[tf.newaxis],
                tf.reshape(tex_coords * map_size, [1, -1, 2]),
                indexing = 'xy')[0]
            shape = tf.concat([self.image_size, [-1]], axis=0)
            albedo = tf.reshape(albedo, shape)

        # interpolate normal vectors
        if normal_map is not None:
            map_size = tf.cast(tf.shape(normal_map)[1::-1], tex_coords.dtype)
            normal_map = tfa.image.interpolate_bilinear(
                normal_map[tf.newaxis],
                tf.reshape(tex_coords * map_size, [1, -1, 2]),
                indexing = 'xy')[0]
            shape = tf.concat([self.image_size, [-1]], axis=0)
            normal_map = tf.reshape(normal_map, shape)
            pixel_normals = normal_map[...,0:1] * tangent1 \
                          + normal_map[...,1:2] * tangent2 \
                          + normal_map[...,2:3] * pixel_normals
            pixel_normals = tf.math.l2_normalize(pixel_normals, axis=-1)

        incoming_ray = light.ray(intersections)
        brdf_diffuse = lambertian.brdf(
            incoming_ray, outgoing_ray, pixel_normals, albedo)
        brdf = brdf_diffuse
        if trimesh.specular_weight > 0.:
            brdf_specular = phong.brdf(
                incoming_ray, outgoing_ray, pixel_normals, shininess, albedo)
            brdf = (1. - trimesh.specular_weight) * brdf_diffuse + trimesh.specular_weight * brdf_specular
        else:
            brdf = brdf_diffuse
        cosine_term = tf.reduce_sum(pixel_normals * -incoming_ray, axis=-1, keepdims=True)
        cosine_term = tf.math.maximum(cosine_term, 0.)
        irradiance = cosine_term * light.intensity(intersections)
        ambient = ambient_light * albedo
        image = brdf * irradiance + ambient

        image = tf.reverse(image, axis=[0])
        valid = tf.reverse(valid, axis=[0])

        if vertex_properties is None:
            return image, valid
        else:
            vertex_properties = tf.reverse(vertex_properties, axis=[0])
            return image, vertex_properties, valid
