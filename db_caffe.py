from StringIO import StringIO
from caffe2.python import core, utils, workspace
from caffe2.proto import caffe2_pb2
import skimage
import skimage.io as io
import skimage.transform
import numpy as np
import glob


def write_caffe_db(db_type, db_name, folder, filetype):
    db = core.C.create_db(db_type, db_name, core.C.Mode.write)
    transaction = db.new_transaction()
                
    i = 0
    filename = folder + '/*.' + filetype
    for filename in glob.glob(filename): #assuming gif
        im = skimage.img_as_float(skimage.io.imread(filename)).astype(np.float32)
        img256 = skimage.transform.resize(img, w_size, y_size))
        img256 = img256.swapaxes(1, 2).swapaxes(0, 1)
        img256 = img256[(2, 1, 0), :, :]
        lab = np.array(0, dtype = "int64")
        feature_and_label = caffe2_pb2.TensorProtos()
        feature_and_label.protos.extend([
            utils.NumpyArrayToCaffe2Tensor(img256),
            utils.NumpyArrayToCaffe2Tensor(lab)
            ])
        transaction.put(
                'train_%03d'.format(i),
                feature_and_label.SerializeToString())
        i += 1
    del transaction
    del db
