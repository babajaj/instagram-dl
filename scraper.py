
import numpy as np

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


# def recent_100_pics(username):
#     """With the input of an account page, scrape the 100 most recent posts urls"""
#     url = "https://www.instagram.com/" + username + "/"
#     browser = Chrome()
#     browser.get(url)
#     post = 'https://www.instagram.com/p/'
#     post_links = []
#     i = 0
#     while len(post_links) < 25:
#         print(i)
#         i+=1
#         links = [a.get_attribute('href') for a in browser.find_elements_by_tag_name('a')]
#         for link in links:
#             if post in link and link not in post_links:
#                 post_links.append(link)
#         scroll_down = "window.scrollTo(0, document.body.scrollHeight);"
#         browser.execute_script(scroll_down)
#     else:
#         return post_links[:25]



with open('profiles.txt', 'w') as filehandle:
    for listitem in profiles:
        filehandle.write('%s\n' % listitem)

    