import tensorflow as tf
import numpy as np

def load_obj(obj_file, convert_to_triangles = True):
    with open(obj_file, 'r') as f:
        vertices = []
        texture_coords = []
        normals = []
        face_v = []
        face_vt = []
        face_vn = []
        lineno = 0

        while True:
            lineno += 1
            line = f.readline()
            if not line:
                break

            cols = line.split()
            if cols[0] == 'v':
                vertices.append(list(map(float, cols[1:4])))
            elif cols[0] == 'vt':
                texture_coords.append(list(map(float, cols[1:4])))
            elif cols[0] == 'vn':
                normals.append(list(map(float, cols[1:4])))
            elif cols[0] == 'f':
                n = len(cols[1:])
                v = []
                vt = []
                vn = []
                for col in cols[1:]:
                    col = list(col.split('/'))
                    v.append(col[0])
                    if len(col) >= 2 and col[1]:
                        vt.append(col[1])
                    if len(col) >= 3 and col[2]:
                        vn.append(col[2])
                if len(v) != n:
                    raise ValueError(f'Invalid line.\n{lineno}: {line}')
                v = map(int, v)
                v = map(lambda x: x-1 if x >= 0 else x + len(vertices), v)
                v = list(v)
                if len(vt) == n:
                    vt = map(int, vt)
                    vt = map(lambda x: x-1 if x >= 0 else x + len(vt), vt)
                    vt = list(vt)
                else:
                    vt = None
                if len(vn) == n:
                    vn = map(int, vn)
                    vn = map(lambda x: x-1 if x >= 0 else x + len(vn), vn)
                    vn = list(vn)
                else:
                    vn = None
                if convert_to_triangles:
                    for i in range(1, n-1):
                        face_v.append([v[0], v[i], v[i+1]])
                        face_vt.append([vt[0], vt[i], vt[i+1]] if vt else None)
                        face_vn.append([vn[0], vn[i], vn[i+1]] if vn else None)
                else:
                    face_v.append(v)
                    face_vt.append(vt)
                    face_vn.append(vn)
        outputs = {
            'vertices': np.array(vertices, dtype=np.float32),
            'vertex_faces': np.array(face_v, dtype=np.int32)
        }
        if any(x is None for x in face_vt):
            if not all(x is None for x in face_vt):
                print('Warning: Texture coordinates are missing for some faces. Omitted.')
        else:
            outputs['texture_coordinates'] = np.array(texture_coords, dtype=np.float32)
            outputs['texture_faces'] = np.array(face_vt, dtype=np.int32)
        if any(x is None for x in face_vn):
            if not all(x is None for x in face_vn):
                print('Warning: Normal vectors are missing for some faces. Omitted.')
        else:
            outputs['normal_vectors'] = np.array(normals, dtype=np.float32)
            outputs['normal_faces'] = np.array(face_vn, dtype=np.int32)
        return outputs

class Trimesh(object):
    def __init__(self, vertices, faces, face_groups=None,
                 albedo = tf.constant([0.5, 0.5, 0.5]),
                 texture_map = None,
                 normal_map = None,
                 texture_coords = None,
                 texture_faces = None,
                 specular_weight = tf.constant(0.3),
                 shininess = tf.constant([1.5]),
                 vertex_properties = None):
        self.vertices = vertices
        self.faces = faces
        if face_groups is None:
            face_groups = [tf.range(faces.shape[0])]
        self.face_groups = face_groups

        if texture_map is None:
            self.texture_map = None
        else:
            self.texture_map = tf.reverse(texture_map, axis=[0])
        if normal_map is None:
            self.normal_map = None
        else:
            self.normal_map = tf.reverse(normal_map, axis=[0])
        self.texture_coords = texture_coords
        self.texture_faces = texture_faces

        self.albedo = albedo
        self.specular_weight = specular_weight
        self.shininess = shininess
        self.vertex_properties = vertex_properties

    @staticmethod
    def from_obj(obj_file, texture_file=None, normal_file=None):
        mesh = load_obj(obj_file)
        if texture_file is None:
            texture = None
        else:
            texture = tf.image.decode_image(tf.io.read_file(texture_file), dtype=tf.float32)
            texture = tf.cast(texture[:,:,:3], tf.float32)
        if normal_file is None:
            normal = None
        else:
            normal = tf.image.decode_image(tf.io.read_file(normal_file), dtype=tf.float32, channels=3)
            normal = normal * 2. - 1.

        trimesh = Trimesh(mesh['vertices'], mesh['vertex_faces'],
                    texture_map = texture,
                    normal_map = normal,
                    texture_coords = mesh.get('texture_coordinates'),
                    texture_faces = mesh.get('texture_faces'))
        return trimesh
