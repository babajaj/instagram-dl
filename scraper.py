
import numpy as np
from PIL import Image
import os
import json
#   list of profiles of length 103. We tried to somewhat balance male/female
#   accounts and get some but not too many types of accounts .
profiles = ["arianagrande", "therock", "kyliejenner", "selenagomez", "kimkardashian", "beyonce", "justinbieber", "natgeo", "kendalljenner",
            "taylorswift", "jlo", "nickiminaj", "khloekardashian", "nike", "mileycyrus", "katyperry", "kourtneykardashians", "kevinhart4real",
            "ddlovato", "theellenshow", "badgalriri", "virat.kohli", "zendaya", "iamcardib", "kingjames", "chrisbrownofficial", "champagnepapi", "billieeilish",
            "shakira", "victoriassecret", "vindiesel", "championsleague", "davidbeckham", "nasa", "gigihadid", "justintimberlake", "emmawatson",
            "priyankachopra", "shawnmendes", "shraddhakapoor", "maluma", "snoopdogg", "dualipa", "9gag", "nba", "camila_cabello", "willsmith",
            "aliaabhatt", "anitta", "marvel", "nehakakkar", "hudabeauty", "robertdowneyjr", "leonardodicaprio", "gal_gadot", "katrinakaif", "chrishemsworth",
            "ladygaga", "JBalvin", "zacefron", "michelleobama", "caradelevingne", "garethbale11", "nikefootball", "lelepons", "premierleague", "gucci",
            "natgeotravel", "dishapatani", "adele", "brunamarquezine", "milliebobbybrown", "vanessahudgens", "vancityreynolds", "tomholland2013", "danbilzerian",
            "jenniferaniston", "shaymitchell", "blakelively", "amandacerny", "camerondallas", "lizakoshy", "jamescharles", "hillaryduff", "jakepaul", 
            "jenselter", "nashgrier", "lilly", "davidmichigan", "joerogan", "gordongram", "simeonpanda", "rickeythompson", "doctor.mike", "justinptrudeau",
            "chrisburkard", "thebodycoach", "emilyskyefit", "thebucketlistfamily", "minimalistbaker", "demibagby", "thesharkdaymond", "meghanrienks"]



# #run this to get the profiles:
# with open('profiles.txt', 'w') as filehandle:
#     for listitem in profiles:
#         filehandle.write('%s\n' % listitem)

img_size = (256, 256)

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
            image = np.asarray(image.resize(img_size)).tolist()
            f = open(folder + '/' + file + '.txt', encoding="utf8")
            caption = f.read().replace("\n", " ")
            f.close
            images.append(image)
            captions.append(caption)
        names.append(name)
    return names, images, captions 


names, images, captions = load_data(profiles[0:3])

print(len(images), len(captions))


def jsonify(names, images, captions):
    with open("data/images.json", "w") as imgs:
        json.dump(images, imgs)
    with open("data/captions.json", "w") as caps:
        json.dump(captions, caps)


jsonify(names, images, captions)

#SCRAPER:
# read first n characters (150)
# resize images


#PARSER:
# tokenize captions by character (emojis, hashtags included)
# build vocab out of tokenizing

