import os
import time
import random
import pygame
import cv2
import mediapipe as mp
import numpy as np
import pickle
# from ML_script import Knn
with open('knn.pickle', 'rb') as f:
    clf = pickle.load(f)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)


WIDTH = 800
HEIGHT = 312
FPS = 30
frameCount = 0
speed = 10
UPFORCE = 30
Gravity = 5


pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dinosaur Game")
clock = pygame.time.Clock()
myfont = pygame.font.Font(None, 20)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


screen.fill(BLACK)


running = True
# count=0
# start = time.time()

all_sprites = pygame.sprite.Group()
game_over = False
# set up asset folders
game_folder = os.path.dirname(__file__)
img_folder = os.path.join(game_folder, 'img')
player_img1 = pygame.image.load(os.path.join(img_folder, 'long1.png'))
player_img2 = pygame.image.load(os.path.join(img_folder, 'long2.png'))
bg_img = pygame.image.load(os.path.join(img_folder, 'map2.png')).convert()
cactus_img1 = pygame.image.load(os.path.join(img_folder, 'cactus01.png'))
cactus_img2 = pygame.image.load(os.path.join(img_folder, 'cactus02.png'))
cactus_img3 = pygame.image.load(os.path.join(img_folder, 'cactus03.png'))


player_img = [player_img1, player_img2]
cactus_img = [cactus_img1, cactus_img2, cactus_img3]


class Player(pygame.sprite.Sprite):
    # sprite for the Player
    def __init__(self):
        # this line is required to properly create the sprite
        pygame.sprite.Sprite.__init__(self)
        # create a plain rectangle for the sprite image
        self.img_count = 0
        self.image = player_img[self.img_count]
        # find the rectangle that encloses the image
        self.rect = self.image.get_rect()
        self.rect.width *= 0.8
        self.rect.height *= 0.8
        # center the sprite on the screen
        self.rect.center = (60, 240)
        self.speed = pygame.math.Vector2(0, 0)

    def update(self):
        # any code here will happen every time the game loop updates
        # self.rect.x += 5
        if frameCount % 5 == 0:
            self.img_count += 1
            self.image = player_img[self.img_count % 2]

        self.rect.y += self.speed.y

        # if(self.speed.y<0):
        self.speed.y += Gravity
        if self.rect.centery > 240:
            self.speed.y = 0
            self.rect.centery = 240

    def upForce(self):
        self.speed.y = -UPFORCE


class Cactus(pygame.sprite.Sprite):
    # sprite for the Player
    def __init__(self, posx):
        # this line is required to properly create the sprite
        pygame.sprite.Sprite.__init__(self)
        # create a plain rectangle for the sprite image
        # self.img_count=0
        self.image = cactus_img[random.randint(0, 2)]
        # find the rectangle that encloses the image
        self.rect = self.image.get_rect()
        self.rect.width *= 0.8
        self.rect.height *= 0.8
        # center the sprite on the screen
        self.rect.center = (posx, 245)
        self.speed = pygame.math.Vector2(-speed, 0)

    def update(self):
        # any code here will happen every time the game loop updates
        self.rect.x += self.speed.x
        if self.rect.centerx < -30:
            del self

        # if self.rect.left > WIDTH:
        #     self.rect.right = 0


player = Player()
all_sprites.add(player)
cactus_group = pygame.sprite.Group()
bg_rect = bg_img.get_rect()
bg_rect.center = screen.get_rect().center
st = time.time()

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                a = hand_landmarks.landmark
                # print(type(a))
                x = []
                y = []
                z = []
                for p in a:
                    x.append(p.x)
                    y.append(p.y)
                    z.append(p.z)
                # print((np.std(x)+np.std(y)),np.mean(z)**2,player.rect.centery)
                predict_class = clf.predict(
                    [[np.mean(x), np.mean(y), np.mean(z), np.std(x), np.std(y)]])
                print(predict_class)
                if (predict_class == 1 and player.rect.centery >= 240):
                    # print(clf.predict(
                    #     [[np.mean(x), np.mean(y), np.mean(z), np.std(x), np.std(y)]]))
                    player.upForce()
                    # print(player.speed.y)
                    # result = model(mean(x),mean(y),mean(z),np.std(x),np.std(y))
                    # A = np.mean(x)
                    # B = np.mean(y)
                    # C = np.mean(z)
                    # D = np.std(x)
                    # E = np.std(y)

                    " cette partie etait pour la récolte de données"
                    # closeHand = np.concatenate(
                    #     [closeHand, np.array([[2, A, B, C, D, E]])], axis=0)

                    # closeHandpd = pd.DataFrame(
                    #     closeHand, columns=['Hand', 'MeanX', 'MeanY', 'MeanZ', 'StdX', 'StdY'])

                    # if (closeHandpd.shape[0] > 250):
                    #     closeHandpd.to_csv("closeHandpd.csv")

                    # result = model(mean(x),mean(y),mean(z),np.std(x),np.std(y))
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.resize(
            cv2.flip(image, 1), (360, 240)))
        if cv2.waitKey(5) & 0xFF == 27:
            break
        if running == False:
            break

        for event in pygame.event.get():
            # check for closing window
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                print(pygame.mouse.get_pos())
            if event.type == pygame.MOUSEBUTTONDOWN and game_over:
                game_over = False
                frameCount = 0
                # del cactus_group
                cactus_group.empty()
                all_sprites.update()
                cactus_group.update()
                screen.blit(bg_img, bg_rect)
                screen.blit(bg_img, (bg_rect.x+bg_rect.width, bg_rect.y))
                cactus_group.draw(screen)
                all_sprites.draw(screen)
                pygame.display.flip()

        screen.blit(bg_img, bg_rect)
        screen.blit(bg_img, (bg_rect.x+bg_rect.width, bg_rect.y))
        if not game_over:
            frameCount += 1
            # print(frameCount/(time.time()-st))
            if pygame.sprite.spritecollide(player, cactus_group, False, pygame.sprite.collide_rect):
                game_over = True
            # add new cactus
            if (frameCount % 15 == 0):
                cactus = Cactus(WIDTH+random.randint(-40, 40))
                cactus_group.add(cactus)
            bg_rect.centerx -= speed
            if (bg_rect.centerx < -WIDTH/2):
                bg_rect.centerx += WIDTH
            all_sprites.update()
            cactus_group.update()
            text2 = myfont.render("Score:"+str(frameCount), True, (0, 0, 0))
            textRect2 = text2.get_rect()
            textRect2.center = (WIDTH-50, 20)
            screen.blit(text2, textRect2)
        else:
            text1 = myfont.render("Game Over", True, (0, 0, 0))
            text3 = myfont.render("Click to restart", True, (0, 0, 0))
            text2 = myfont.render("Score:"+str(frameCount), True, (0, 0, 0))
            textRect1 = text1.get_rect()
            textRect2 = text2.get_rect()
            textRect3 = text3.get_rect()
            textRect1.center = (WIDTH/2, 100)
            textRect2.center = (WIDTH/2, 150)
            textRect3.center = (WIDTH/2, 200)
            screen.blit(text1, textRect1)
            screen.blit(text2, textRect2)
            screen.blit(text3, textRect3)

        cactus_group.draw(screen)
        all_sprites.draw(screen)
        pygame.display.flip()

cap.release()
pygame.quit()
