class ConfigSimple(object):
    num_images = 1
    image = 'Training_images/image6.jpg'
    total_rows = 200
    total_cols = 200

    num_episodes = 10000
    max_steps = 175


    lr = 0.00001
    batch_size = 1
    sight_dim = 2
    seq_length = 6
    vision_size = (sight_dim * 2 + 1) * (sight_dim * 2 + 1)
    discount_factor = 0.99

    plot_interval = 100

    save_dir = 'Mining_Saves/Test20'

    img_dir = 'Training_images'