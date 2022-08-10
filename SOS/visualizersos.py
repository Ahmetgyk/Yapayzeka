import cv2
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import ayarlar
import copy
from sos1 import SosEnv

class Visualizer:


    def __init__(self, sizeDiv = 6):
        self.env = None
        self.selected = False
        self.size = 1800
        self.spacing = 1800/sizeDiv
        self.font = ImageFont.truetype('arial.ttf', 275)

        self.emptyBoard = self.drawEmptyBoard()


    def drawEmptyBoard(self):


        image = Image.new("RGBA", (self.size, self.size))
        drawer = ImageDraw.Draw(image, 'RGBA')
        for i in range(1,7):
            drawer.line([(self.spacing*i,0), (self.spacing*i, self.size)], fill=None, width=10)
            drawer.line([(0,self.spacing*i), (self.size, self.spacing*i)], fill=None, width=10)
        return image

        

    def drawPiecesOnBoard(self):
        boardImage = copy.deepcopy(self.emptyBoard)
        draw = ImageDraw.Draw(boardImage)
        

        for i, piece in enumerate(self.env.board):
            if piece == 1:
                draw.text((i%6 *self.spacing + self.spacing//5, i//6 * self.spacing),"O",(255,255,255),font=self.font)
                
            elif piece == -1:
                draw.text((i%6 *self.spacing + self.spacing//5 , i//6 * self.spacing),"S",(255,255,255),font=self.font)
                

        return boardImage


    def show(self, waitTime = ayarlar.WAIT_TIME):
        self.open_cv_image = np.array(self.drawPiecesOnBoard())
        cv2.namedWindow('Board', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Board', 800,800)
        cv2.imshow('Board', self.open_cv_image)
        cv2.waitKey(waitTime)