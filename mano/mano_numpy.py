import os

import numpy as np
import pickle

from cv2 import Rodrigues as rodrigues

LEFT_PATH = "MANO_LEFT.pkl"
RIGHT_PATH = "MANO_RIGHT.pkl"

class MANO():
    def __init__(self, left, model_dir):
        super(MANO, self).__init__()

        model_paths = {
            'left': os.path.join(model_dir, LEFT_PATH),
            'right': os.path.join(model_dir, RIGHT_PATH)
        }

        with open(model_paths[left], 'rb') as f:
            mano_model = pickle.load(f, encoding='latin1')
        self.J_regressor = np.array(mano_model['J_regressor'].todense())
        self.weights = mano_model['weights']
        self.posedirs = mano_model['posedirs']
        self.v_template = mano_model['v_template']
        self.shapedirs = np.array(mano_model['shapedirs'])
        # self.faces = mano_model['f'].astype('int32')
        self.kintree_table = mano_model['kintree_table'].astype('int64')

        id_to_col = {self.kintree_table[1, i].item(): i for i in range(self.kintree_table.shape[1])}
        self.parent = np.array([id_to_col[self.kintree_table[0, it]] for it in range(1, self.kintree_table.shape[1])])

        self.pose_shape = [16, 3]
        self.beta_shape = [10]
        self.pose = np.zeros(self.pose_shape)
        self.beta = np.zeros(self.beta_shape)

        self.verts = None
        self.J = None
        self.R = None

    def __call__(self, pose, beta):

        v_template = self.v_template              # (6890, 3)
        shapedirs = self.shapedirs.reshape(-1,10) # (6890, 10)
        beta = beta[:, None]                      # (10, 1)

        v_shaped = shapedirs.dot(beta).reshape(778, 3) + v_template # (6890, 3)
        J = self.J_regressor.dot(v_shaped)                           # (24, 3)

        # input is a rotation matrix: (24,3,3)
        if pose.shape == (16, 3, 3):
            R = pose
        # input is a rotation axis-angle vector: (1, 72), (72, 1) or (72, )
        elif pose.shape == (1, 48) or pose.shape == (48, 1) or pose.shape == (48,):
            pose_vectors = pose.reshape(-1, 3)                      # (24, 3)
            R = np.array([rodrigues(pose_vectors[p_idx])[0] 
                            for p_idx in range(pose_vectors.shape[0])
                          ], 
                          dtype='float32')                          # (24, 3, 3)
        else:
            raise ValueError("Unsupported Pose Inputs - the Pose Shape is {}".format(pose.shape))
        
        Is = np.eye(3, dtype='float32')[None, :]                    # (1, 3, 3)
        lrotmin = (R[1:,:] - Is).reshape(-1, 1)                     # (23x3x3, 1)
        posedirs = self.posedirs.reshape(-1,135)                    # (6890, 207)
        v_posed = v_shaped + posedirs.dot(lrotmin).reshape(778, 3) # (6890, 3)

        J_ = J.copy()
        J_[1:, :] = J[1:, :] - J[self.parent, :]                     # (24, 3)
        G_ = np.concatenate([R, J_[:, :, None]],  axis=-1)           # (24, 3, 4)
        pad_rows = np.array([[0, 0, 0, 1]], dtype='float32')
        pad_rows = np.repeat(pad_rows, 16, axis=0).reshape(-1, 1, 4)
        G_ = np.concatenate([G_, pad_rows], axis=1)                  # (24, 4, 4)

        G = [G_[0].copy()]
        for i in range(1, 16):
            G.append(G[self.parent[i-1]].dot(G_[i, :, :]))
        G = np.stack(G, axis=0)  # (24, 4, 4)

        joints = G[:, :3, 3]

        rest_joints = np.concatenate([J, np.zeros((16, 1))], axis=-1)[:, :, None]  # (24, 4, 1)
        zeros = np.zeros((16, 4, 3), dtype='float32')                              # (24, 4, 3)
        rest_joints_mtx = np.concatenate([zeros, rest_joints], axis=-1)            # (24, 4, 4) 
        posed_joints_mtx = np.matmul(G, rest_joints_mtx)
        G = G - posed_joints_mtx
                                                            
        rest_shape_h = np.concatenate([v_posed, np.ones(v_posed.shape[0])[:, None]], axis=-1) #(6890, 4)
        T = self.weights.dot(G.reshape(16, -1)).reshape(778, 4, 4)
        v = np.matmul(T, rest_shape_h[:, :, None])[:, :3, 0]

        # # Apply the translation to the joint positions
        # joints += handTrans

        # # Apply the translation to the vertex positions
        # v += handTrans

        
        return v, joints
        