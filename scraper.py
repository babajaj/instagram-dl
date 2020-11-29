
import numpy as np
from PIL import Image
import os
import json
#   list of profiles of length 103. We tried to somewhat balance male/female
#   accounts and get some but not too many types of accounts .
profiles = ["arianagrande", "therock", "kyliejenner", "selenagomez", "kimkardashian", "beyonce", "justinbieber", "natgeo", "kendalljenner",
            "taylorswift", "jlo", "nickiminaj", "khloekardashian", "nike", "mileycyrus", "katyperry", "kourtneykardash", "kevinhart4real",
            "ddlovato", "theellenshow", "badgalriri", "virat.kohli", "zendaya", "iamcardib", "kingjames", "chrisbrownofficial", "champagnepapi", "billieeilish",
            "shakira", "victoriassecret", "vindiesel", "championsleague", "davidbeckham", "nasa", "gigihadid", "justintimberlake", "emmawatson",
            "priyankachopra", "shawnmendes", "shraddhakapoor", "snoopdogg", "dualipa", "9gag", "nba", "camila_cabello", "willsmith",
            "aliaabhatt", "marvel", "nehakakkar", "hudabeauty", "robertdowneyjr", "leonardodicaprio", "gal_gadot", "katrinakaif", "chrishemsworth",
            "ladygaga", "zacefron", "michelleobama", "caradelevingne", "garethbale11", "nikefootball", "lelepons", "premierleague", "gucci",
            "natgeotravel", "dishapatani", "adele", "brunamarquezine", "milliebobbybrown", "vanessahudgens", "vancityreynolds", "tomholland2013", "danbilzerian",
            "jenniferaniston", "shaymitchell", "blakelively", "amandacerny", "camerondallas", "lizakoshy", "jamescharles", "hillaryduff", "jakepaul", 
            "jenselter", "nashgrier", "lilly", "davidmichigan", "joerogan", "gordongram", "simeonpanda", "rickeythompson", "doctor.mike", "justinptrudeau",
            "chrisburkard", "thebodycoach", "emilyskyefit", "thebucketlistfamily", "minimalistbaker", "demibagby", "thesharkdaymond", "meghanrienks"]



profiles2 = profiles[0:3]


# #run this to get the profiles:
# with open('profiles.txt', 'w') as filehandle:
#     for listitem in profiles:
#         filehandle.write('%s\n' % listitem)

img_size = (224, 224)

def get_file_names(path):
    txt_files = []
    arr = os.listdir(path)
    for item in arr:
        if item.endswith('.txt'):
            file_name = item.split('.')[0]
            txt_files.append(file_name)
    return txt_files

def load_data(profiles):
    folder_index = 0
    images = []
    captions = []
    names = []
    for name in profiles:
        folder = "data/" + name
        file_names = get_file_names(folder)
        for file in file_names:
            try:
                image = Image.open(folder + '/' + file + '.jpg')
            except Exception:
                image = Image.open(folder + '/' + file + '_1.jpg')
            
            image = np.asarray(image.resize(img_size))
            

            f = open(folder + '/' + file + '.txt', encoding="utf8")
            caption = f.read().replace("\n", " ")
            f.close
            if (len(image.shape) != 2):
                images.append(image)
                captions.append(caption)
        names.append(name)
    return names, np.array(images), captions 


names, images, captions = load_data(profiles2)

print(len(images), len(captions))

def pickle(names, images, captions):
    with open("data/images.npy", "wb") as imgs:
        np.save(imgs, images, allow_pickle=True)
    print("imgs done")
    with open("data/captions.json", "w") as caps:
        json.dump(captions, caps)
    print('captions done')

pickle(names, images, captions)

print("done with making a pickle")

#SCRAPER:
# read first n characters (150)
# resize images


#PARSER:
# tokenize captions by character (emojis, hashtags included)
# build vocab out of tokenizing

