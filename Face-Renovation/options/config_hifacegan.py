import sys

class TrainOptions(object):
    dataroot = '<your dataset folder, [LQ;HQ]>'
    dataroot_assist = ''
    name = '<your experiment name>'
    crop_size = 512

    gpu_ids = [0]  # set to [] for CPU-only training (not tested)
    gan_mode = 'ls'

    continue_train = False
    which_epoch = 'latest'

    D_steps_per_G = 1
    aspect_ratio = 1.0
    batchSize = 2
    beta1 = 0.0
    beta2 = 0.9
    cache_filelist_read = True
    cache_filelist_write = True
    checkpoints_dir = './checkpoints'
    choose_pair = [0, 1]
    coco_no_portraits = False
    contain_dontcare_label = False

    dataset_mode = 'train'
    debug = False
    display_freq = 100
    display_winsize = 256
    print_freq = 100
    save_epoch_freq = 1
    save_latest_freq = 5000

    init_type = 'xavier'
    init_variance = 0.02
    isTrain = True
    is_test = False

    semantic_nc = 3
    label_nc = 3
    output_nc = 3
    lambda_feat = 10.0
    lambda_kld = 0.05
    lambda_vgg = 10.0
    load_from_opt_file = False
    lr = 0.0002
    max_dataset_size = sys.maxsize
    model = 'pix2pix'
    nThreads = 2

    n_layers_D = 4
    num_D = 2
    ndf = 64
    nef = 16
    netD = 'multiscale'
    netD_subarch = 'n_layer'
    netG = 'hifacegan'  # spade, lipspade
    ngf = 64  # set to 48 for Titan X 12GB card
    niter = 30
    niter_decay = 20
    no_TTUR = False
    no_flip = False
    no_ganFeat_loss = False
    no_html = False
    no_instance = True
    no_pairing_check = False
    no_vgg_loss = False

    norm_D = 'spectralinstance'
    norm_E = 'spectralinstance'
    norm_G = 'spectralspadesyncbatch3x3'

    num_upsampling_layers = 'normal'
    optimizer = 'adam'
    phase = 'train'
    prd_resize = 512
    preprocess_mode = 'resize_and_crop'

    serial_batches = False
    tf_log = False
    train_phase = 3  # progressive training disabled (set initial phase to 0 to enable it)
    # 20200211
    #max_train_phase = 2 # default 3 (4x)
    max_train_phase = 3
    # training 1024*1024 is also possible, just turning this to 4 and add more layers in generator.
    upsample_phase_epoch_fq = 5
    use_vae = False
    z_dim = 256


class TestOptions(object):
    dataroot = '/nfs/home_new/gjffl357/realimages_256'
    name = '4xsr'
    results_dir = './results/real'
    gpu_ids = [0]
    crop_size = 512
    dataset_mode = 'test'
    which_epoch = 'latest'

    aspect_ratio = 1.0
    batchSize = 1
    cache_filelist_read = True
    cache_filelist_write = True
    checkpoints_dir = './checkpoints'
    coco_no_portraits = False
    contain_dontcare_label = False

    display_winsize = 256
    how_many = sys.maxsize
    #how_many = 10
    init_type = 'xavier'
    init_variance = 0.02
    isTrain = False
    is_test = True
    label_nc = 3
    output_nc = 3
    semantic_nc = 3
    load_from_opt_file = False
    max_dataset_size = sys.maxsize
    
    # make sure the following options match the TrainOptions
    model = 'pix2pix'
    nThreads = 0
    netG = 'hifacegan'
    # netG = 'lipspade'
    # nef = 16
    ngf = 48
    no_flip = True
    no_instance = True
    no_pairing_check = False

    norm_D = 'spectralinstance'
    norm_E = 'spectralinstance'
    norm_G = 'spectralspadesyncbatch3x3'
    num_upsampling_layers = 'normal'
    phase = 'test'
    prd_resize = 512
    preprocess_mode = 'resize_and_crop'
    serial_batches = True
    use_vae = False
    z_dim = 256
