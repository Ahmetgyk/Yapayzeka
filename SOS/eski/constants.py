from PIL import Image, ImageDraw


#Görüntüleme ile alakalı parametreler
SIZE = 1600
SPACING = SIZE//3

IS_VISUALIZER_ON = False
WAIT_TIME = 200 #0 girilir ise elle devam ettiriliyor
SHOW_EVERY = 10000


#Ödül/Ceza
WIN_REWARD = 1
LOSE_REWARD = -1
DRAW_REWARD = 0

# Exploration settings
epsilon = 1 # not a constant, going to be decayed
EPSILON_DECAY = 0.99999 #0.9996
MIN_EPSILON = 0.01


#Kayıt ile alakalı parametreler
MODEL_NAME = "3x36Linear-1E-0.1L-Tanh-D999996-Tensorflow" #modelin hangi adla kayıt edileceği
AGGREGATE_STATS_EVERY = 1000  # episodes
LOAD_MODEL = False
IS_TEST = False
MODEL_PATH = "3x36Linear-1E-0.1L-Tanh-D999996-Tensorflow_1611603289" # yüklenecek modelin adı

# Diğer parametreler
DISCOUNT = 1
UPDATE_TARGET_EVERY = 5
LR = 0.1
EPISODES = 2_000_000
ACTION_LIST = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]








#X ve O karakterlerinin oluşturulması.
pieces= {(0,0):"",(0,1):"x",(1,0):"o"}

for piece in pieces:
    image = Image.new("RGBA", (SPACING-100, SPACING-100))
    drawer = ImageDraw.Draw(image, 'RGBA')

    if pieces[piece] == 'x':
        drawer.line([(0,0), (image.size[0], image.size[0])], fill=None, width=10)
        drawer.line([(0,image.size[0]), (image.size[0], 0)], fill=None, width=10)


    elif pieces[piece] == 'o':
        drawer.ellipse((0,0, image.size), outline ='white', width = 10)

    pieces[piece] = image