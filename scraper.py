
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


def load_data(profiles):
    folder_index = 0
    get_image = True
    image = None
    caption = None
    images = []
    captions = []
    names = []
    for name in profiles:
        folder = "data/" + name
        arr = os.listdir(folder)
        for file in arr:
            if get_image:
                image = Image.open(folder + '/' + file)
            else:
                if file.endswith(".txt"):
                    with open(file, 'r') as f:
                        caption = f.read().replace('\n', '')
            folder_index += 1
            images.append(image)
            captions.append(caption)
            names.append(name)
    return names, images, captions 

    


    