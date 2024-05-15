"""Create lmdb dataset"""
from termios import XCASE
from util import *
import lmdb
import caffe
import scipy.io

def create_lmdb_train(
    datadir, fns, name, matkey,
    crop_sizes, scales, ksizes, strides,
    load=h5py.File, augment=True,
    seed=2017,trans=1,norm=0):
    """
    Create Augmented Dataset
    """
    def preprocess(data):
        new_data = []
        data[data < 0] = 0
        data = minmax_normalize(data)
        data = np.rot90(data, k=2, axes=(1,2)) # ICVL
        # data = minmax_normalize(data.transpose((2,0,1))) # for Remote Sensing
        # Visualize3D(data)
        if crop_sizes is not None:
            data = crop_center(data, crop_sizes[0], crop_sizes[1])        
        
        for i in range(len(scales)):
            if scales[i] != 1:
                temp = zoom(data, zoom=(1, scales[i], scales[i]))
            else:
                temp = data
            # print(temp.shape)
            temp = Data2Volume(temp, ksizes=ksizes, strides=list(strides[i]))            
            new_data.append(temp)
        new_data = np.concatenate(new_data, axis=0)
        if augment:
             for i in range(new_data.shape[0]):
                 new_data[i,...] = data_augmentation(new_data[i, ...])
                
        return new_data.astype(np.float32)

    np.random.seed(seed)
    scales = list(scales)
    ksizes = list(ksizes)        
    assert len(scales) == len(strides)
    # calculate the shape of dataset
    data = scipy.io.loadmat(datadir + fns[0])
    data= data[matkey]
    if trans:
        data = np.transpose(data,(2,0,1))
    # print(data.shape)
    data = preprocess(data)
    N = data.shape[0]
    
    # print(data.shape)
    map_size = data.nbytes * len(fns) * 1.2
    print('map size (GB):', map_size / 1024 / 1024 / 1024)
    
    #import ipdb#; ipdb.set_trace()
    if os.path.exists(name+'.db'):
        raise Exception('database already exist!')
    env = lmdb.open(name+'.db', map_size=map_size, writemap=True)
    with env.begin(write=True) as txn:
        # txn is a Transaction object
        k = 0
        for i, fn in enumerate(fns):
            try:
                X = scipy.io.loadmat(datadir + fn)[matkey]
                if trans:
                    # print(X.shape)
                    X = np.transpose(X,(2,0,1))
            except:
                print('loading', datadir+fn, 'fail')
                continue    
            # X = preprocess(X)
            X = data        
            N = X.shape[0]
            for j in range(N):
                # print(X[j].max(), X[j].min())
                if X[j].min() < -100:
                    continue
                elif X[j].max() == 0:
                    continue
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = X.shape[1]
                datum.height = X.shape[2]
                datum.width = X.shape[3]
                # print(X[j].max(), X[j].min())
                if norm == 0:
                    datum.data = X[j].tobytes()
                else:
                    datum.data = minmax_normalize(X[j]).tobytes()
                str_id = '{:08}'.format(k)
                k += 1
                txn.put(str_id.encode('ascii'), datum.SerializeToString())
            print('load mat (%d/%d): %s' %(i,len(fns),fn))
            print(k)

        print('done')


# Create Pavia Centre dataset 
def create_PaviaCentre():
    print('create Pavia Centre...')
    datadir = './data/PaviaCentre/'
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0]+'.mat' for fn in fns]

    create_lmdb_train(
        datadir, fns, '/home/kaixuan/Dataset/PaviaCentre', 'hsi',  # your own dataset address
        crop_sizes=None,
        scales=(1,),
        ksizes=(101, 64, 64),
        strides=[(101, 32, 32)],
        load=loadmat, augment=True,
    )

# Create ICVL training dataset
def create_icvl64_31():
    print('create icvl64_31...')
    datadir = '/nas_data/fugym/ICVL/train/' # your own data address
    fns = os.listdir(datadir) 
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    
    create_lmdb_train(
        datadir, fns, '/nas_data/fugym/ICVL/generated/ICVL64_31', 'rad',  # your own dataset address
        crop_sizes=(1024, 1024),
        scales=(1, 0.5, 0.25),        
        ksizes=(31, 64, 64),
        strides=[(31, 64, 64), (31, 32, 32), (31, 32, 32)],        
        load=h5py.File, augment=True,
    )

def create_cave64_31_2():
    print('create cave64_31_2...')
    datadir = '/nas_data/fugym/datasets/cave_2/' # your own data address
    fns = os.listdir(datadir) 
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    
    create_lmdb_train(
        datadir, fns, '/nas_data/fugym/datasets/CAVE64_31_2', 'DataCube',  # your own dataset address
        crop_sizes=None,
        scales=(1, 0.5, 0.25),        
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32), (31, 16, 16), (31, 16, 16)],     
        load=scipy.io.loadmat, augment=True, trans=1
    )

def create_cave64_31_20():
    print('create cave64_31_2...')
    datadir = '/nas_data/xiongfc/CVPR2022/CAVE/part/train/' # your own data address
    fns = os.listdir(datadir) 
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    
    create_lmdb_train(
        datadir, fns, '/nas_data/fugym/datasets/CAVE64_31_20', 'DataCube',  # your own dataset address
        crop_sizes=None,
        scales=(1, 0.5, 0.25),        
        ksizes=(31, 64, 64),
        strides=[(31, 16, 16), (31, 8, 8), (31, 8, 8)],    
        load=scipy.io.loadmat, augment=True, trans=1
    )

def create_vis64_16():
    print('create vis64_31...')
    datadir = '/home/ironkitty/nas_data/datasets/vis_2/' # your own data address
    fns = os.listdir(datadir) 
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    try:
        create_lmdb_train(
            datadir, fns, '/nas_data/fugym/datasets/vis64_31_2', 'DataCube',  # your own dataset address
            crop_sizes=None,
            scales=(1, 0.5, 0.25),        
            ksizes=(24, 64, 64),
            strides=[(16, 64, 64), (16, 32, 32), (16, 32, 32)],        
            load=h5py.File, augment=True,
        )
    except:
        create_lmdb_train(
            datadir, fns, '/nas_data/fugym/datasets/vis64_31_2', 'DataCube',  # your own dataset address
            crop_sizes=None,
            scales=(1, 0.5, 0.25),        
            ksizes=(16, 64, 64),
            strides=[(16, 64, 64), (16, 32, 32), (16, 32, 32)],        
            load=scipy.io.loadmat, augment=True,trans=1,
        )

def create_nir64_16():
    print('create nir64_31...')
    datadir = '/nas_data/xiongfc/nir_processed/train/' # your own data address
    fns = os.listdir(datadir) 
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    
    create_lmdb_train(
        datadir, fns, '/nas_data/fugym/datasets/nir64_31', 'DataCube',  # your own dataset address
        crop_sizes=None,
        scales=(1, 0.5, 1/3),        
        ksizes=(24, 64, 64),
        strides=[(24, 64, 64), (24, 32, 32), (24, 32, 32)],        
        load=scipy.io.loadmat, augment=True, trans=1
    )

def create_icvl512_31():
    print('create icvl512_31...')
    datadir = '/nas_data/fugym/ICVL/train2_2mats/' # your own data address
    fns = os.listdir(datadir) 
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    
    create_lmdb_train(
        datadir, fns, '/nas_data/fugym/ICVL/generated/ICVL512_31_2mats', 'rad',  # your own dataset address
        crop_sizes=(1024, 1024),
        scales=(1, 0.5, 0.25),        
        ksizes=(31, 512, 512),
        strides=[(31, 256, 256), (31, 256, 256), (31, 256, 256)],        
        load=h5py.File, augment=True,
    )

def create_icvl128_31():
    print('create icvl128_31...')
    datadir = '/nas_data/fugym/ICVL/train_5mats/' # your own data address
    fns = os.listdir(datadir) 
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    
    create_lmdb_train(
        datadir, fns, '/nas_data/fugym/ICVL/generated/ICVL128_31_5', 'rad',  # your own dataset address
        crop_sizes=(1024, 1024),
        scales=(1, 0.5, 0.25),        
        ksizes=(31, 128, 128),
        strides=[(31, 64, 64), (31, 64, 64), (31, 64, 64)],        
        load=h5py.File, augment=True,
    )

def create_icvl256_31():
    print('create icvl256_31...')
    datadir = '/nas_data/fugym/ICVL/train2/' # your own data address
    fns = os.listdir(datadir) 
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    
    create_lmdb_train(
        datadir, fns, '/nas_data/fugym/ICVL/generated/ICVL256_31', 'rad',  # your own dataset address
        crop_sizes=(1024, 1024),
        scales=(1, 0.5, 0.25),        
        ksizes=(31, 256, 256),
        strides=[(31, 128, 128), (31, 128, 128), (31, 128, 128)],        
        load=h5py.File, augment=True,
    )

def create_wdc64_31():
    print('create wdc64_31...')
    datadir = '/home/ironkitty/nas_data/datasets/wdc/train/' # your own data address
    fns = os.listdir(datadir) 
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    
    create_lmdb_train(
        datadir, fns, '/nas_data_fibre/fugym/datasets/wdc/wdc64_31', 'wdc_train',  # your own dataset address
        crop_sizes=None,
        scales=(1, 0.5, 0.25),        
        ksizes=(31, 64, 64),
        strides=[(15, 32, 32), (8, 16, 16), (8, 16, 16)],        
        load=h5py.File, augment=True,
    )

def create_wdc64_191():
    print('create wdc64_191...')
    datadir = '/home/ironkitty/nas_data/datasets/wdc/train/' # your own data address
    fns = os.listdir(datadir) 
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    
    create_lmdb_train(
        datadir, fns, '/nas_data_fibre/fugym/datasets/wdc/wdc64_191', 'wdc_train',  # your own dataset address
        crop_sizes=None,
        scales=(1, 0.5, 0.25),        
        ksizes=(191, 64, 64),
        strides=[(95, 16, 16), (47, 8, 8), (47, 8, 8)],        
        load=h5py.File, augment=True,
    )

def create_wdc64_0_31():
    print('create wdc64_31...')
    datadir = '/home/ironkitty/nas_data/datasets/wdc/train_0/' # your own data address
    fns = os.listdir(datadir) 
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    
    create_lmdb_train(
        datadir, fns, '/nas_data_fibre/fugym/datasets/wdc/wdc64_0_31', 'wdc_train',  # your own dataset address
        crop_sizes=None,
        scales=(1, 0.5, 0.25),        
        ksizes=(31, 64, 64),
        strides=[(15, 32, 32), (8, 16, 16), (8, 16, 16)],        
        load=h5py.File, augment=True,
    )

def create_wdc64_0_191():
    print('create wdc64_191...')
    datadir = '/home/ironkitty/nas_data/datasets/wdc/train_0/' # your own data address
    fns = os.listdir(datadir) 
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    
    create_lmdb_train(
        datadir, fns, '/nas_data_fibre/fugym/datasets/wdc/wdc64_0_191', 'wdc_train',  # your own dataset address
        crop_sizes=None,
        scales=(1, 0.5, 0.25),        
        ksizes=(191, 64, 64),
        strides=[(95, 16, 16), (47, 8, 8), (47, 8, 8)],        
        load=h5py.File, augment=True,
    )

def create_wdc64_stride_24_31():
    print('create wdc64_31...')
    datadir = '/home/ironkitty/nas_data/datasets/wdc/train/' # your own data address
    fns = os.listdir(datadir) 
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    
    create_lmdb_train(
        datadir, fns, '/nas_data_fibre/fugym/datasets/wdc/wdc64_stride_24_31_no0_01', 'wdc_train',  # your own dataset address
        crop_sizes=None,
        scales=(1, 0.5, 0.25),        
        ksizes=(31, 64, 64),
        strides=[(10, 24, 24), (5, 12, 12), (5, 12, 12)],        
        load=h5py.File, augment=True,
    )

def create_houston64_stride_24_46():
    print('create houston64_46...')
    datadir = '/home/ironkitty/nas_data/datasets/houston/train_46/' # your own data address
    fns = os.listdir(datadir) 
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    
    create_lmdb_train(
        datadir, fns, '/home/ironkitty/nas_data/datasets/houston/houston64_stride_24_46', 'houston',  # your own dataset address
        crop_sizes=None,
        scales=(1, 0.5, 0.25),        
        ksizes=(46, 64, 64),
        strides=[(10, 24, 24), (5, 12, 12), (5, 12, 12)],   
        load=h5py.File, augment=True,
    )

def create_houston64_stride_24_46_norm():
    print('create houston64_46_norm...')
    datadir = '/home/ironkitty/nas_data/datasets/houston/train_46/' # your own data address
    fns = os.listdir(datadir) 
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    
    create_lmdb_train(
        datadir, fns, '/home/ironkitty/nas_data/datasets/houston/houston64_stride_24_46_norm', 'houston',  # your own dataset address
        crop_sizes=None,
        scales=(1, 0.5, 0.25),        
        ksizes=(46, 64, 64),
        strides=[(10, 24, 24), (5, 12, 12), (5, 12, 12)],   
        load=h5py.File, augment=True,norm=1
    )

def create_houston512_norm():
    print('create houston512_46_norm...')
    datadir = '/home/ironkitty/nas_data/datasets/houston/train_46/' # your own data address
    fns = os.listdir(datadir) 
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    
    create_lmdb_train(
        datadir, fns, '/home/ironkitty/nas_data/datasets/houston/houston512_46_norm', 'houston',  # your own dataset address
        crop_sizes=None,
        scales=(1, 0.5, 0.25),        
        ksizes=(46, 512, 512),
        strides=[(46, 256, 256), (46, 256, 256), (46, 256, 256)],   
        load=h5py.File, augment=True,norm=1
    )


def create_houston512():
    print('create houston512_46...')
    datadir = '/home/ironkitty/nas_data/datasets/houston/train_46/' # your own data address
    fns = os.listdir(datadir) 
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    
    create_lmdb_train(
        datadir, fns, '/home/ironkitty/nas_data/datasets/houston/houston512_46', 'houston',  # your own dataset address
        crop_sizes=None,
        scales=(1, 0.5, 0.25),        
        ksizes=(46, 512, 512),
        strides=[(46, 256, 256), (46, 256, 256), (46, 256, 256)],   
        load=h5py.File, augment=True,norm=0
    )

if __name__ == '__main__':
    # create_houston64_stride_24_46()
    # create_houston64_stride_24_46_norm()
    create_houston512()
    #create_icvl64_31()
    # create_PaviaCentre()
    pass
