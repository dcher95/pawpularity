# Collection of Functions for running notebooks

### EDA

import matplotlib.pyplot as plt

def pawpularity_pics(df, num_images, desired_pawpularity, random_state):
    '''The pawpularity_pics() function accepts 4 parameters: df is a dataframe, 
    num_images is the number of images you want displayed, desired_pawpularity 
    is the pawpularity score of pics you want to see, and random state ensures reproducibility.'''
    #how many images to display
    num_images = num_images
    #set the rample state for the sampling for reproducibility
    random_state = random_state
    
    #filter the train_df on the desired_pawpularity and use .sample() to get a sample
    random_sample = df[df["Pawpularity"] == desired_pawpularity].sample(num_images, random_state=random_state).reset_index(drop=True)
    
    #The for loop goes as many loops as specified by the num_images
    for x in range(num_images):
        #start from the id in the dataframe
        image_path_stem = random_sample.iloc[x]['Id']
        root = '../input/petfinder-pawpularity-score/train/'
        extension = '.jpg'
        image_path = root + str(image_path_stem) + extension
         
        #get the pawpularity to confirm it worked
        pawpularity_by_id = random_sample.iloc[x]['Pawpularity']
    
        #use plt.imread() to read in the image file
        image_array = plt.imread(image_path)
        #make a subplot space that is 1 down and num_images across
        plt.subplot(1, num_images, x+1)
        #title is the pawpularity score from the id
        title = pawpularity_by_id
        plt.title(title) 
        #turn off gridlines
        plt.axis('off')
        #then plt.imshow() can display it for you
        plt.imshow(image_array)
    plt.show()
    plt.close()
    
### Data Augmentation
class config:
    DIRECTORY_PATH = "../input/petfinder-pawpularity-score"
    TRAIN_FOLDER_PATH = DIRECTORY_PATH + "/train"
    TRAIN_CSV_PATH = DIRECTORY_PATH + "/train.csv"
    TEST_CSV_PATH = DIRECTORY_PATH + "/test.csv"
    
    SEED = 42
    
# wandb config
WANDB_CONFIG = {
     'competition': 'PetFinder', 
              '_wandb_kernel': 'neuracort'
    }

def set_seed(seed=config.SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed) # Sets the seed for generating random numbers
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Documentation regarding randomness in PyTorch devices: https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True # only applies to CUDA convolution operations 
    
    # It enables benchmark mode in cudnn. benchmark mode is good whenever your input sizes for your network do not vary. This way, cudnn will look for the optimal set of algorithms for that particular configuration (which takes some time). This usually leads to faster runtime.
    # But if your input sizes changes at each iteration, then cudnn will benchmark every time a new size appears, possibly leading to worse runtime performances.
    torch.backends.cudnn.benchmark = False