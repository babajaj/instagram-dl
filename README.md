
By: Eyal Levin and Tali Bers

## Introduction:

Despite the popularity of Instagram, instagram captions are confusing and often have nothing to do with the photo. Some captions “make sense” for the photo, and others don’t, and for humans this distinction can be pretty obvious in many cases. so we set out on a journey to decode instagram captions. In this project, we attempted to train a caption generator on image/caption pairs, and generate captions for our favorite photos. The goal is not to output the “correct” caption, but one that could work for the photo. 

## Data:

We got our data directly from instagram using instalooter, a program that downloads instagram pictures without any API access. We manually ran instalooter on 58 accounts to get around 100 of their most recent posts where the first picture wasn’t a video. Posts can include up to ten pictures/videos and a caption. The 58 accounts are all verified accounts with lots of posts and followers there is a balance between males and females and although most are personal accounts there are a few that are of sports teams or other brands. These are the accounts we scraped from.


["arianagrande", "therock", "kyliejenner", "selenagomez", "kimkardashian", "beyonce", "justinbieber", "natgeo", "kendalljenner",
           "taylorswift", "jlo", "nickiminaj", "khloekardashian", "nike", "mileycyrus", "katyperry", "kourtneykardash", "kevinhart4real",
           "ddlovato", "theellenshow", "badgalriri", "virat.kohli", "zendaya", "iamcardib", "kingjames", "chrisbrownofficial", "champagnepapi", "billieeilish",
           "shakira", "victoriassecret", "vindiesel", "championsleague", "davidbeckham", "nasa", "gigihadid", "justintimberlake", "emmawatson",
           "priyankachopra", "shawnmendes", "shraddhakapoor", "snoopdogg", "dualipa", "9gag", "nba", "camila_cabello", "willsmith",
           "aliaabhatt", "marvel", "nehakakkar", "hudabeauty", "robertdowneyjr", "leonardodicaprio", "gal_gadot", "katrinakaif", "chrishemsworth",
           "ladygaga", "zacefron", "michelleobama"]


We then preprocess the data by taking the first picture from every post that has a caption and running the pictures through a VGG to extract the features. We tokenized all the captions by character to ensure that the emojis, hashtags, and tags could all be reproduced. We unked characters that appear less than 50 times to make sure that we were only learning relevant ones. Finally, we made sure each caption was 150 characters long by either cutting it off or adding padding, each caption had a start and stop character as well. 

## Methodology:

We followed an architecture based on Marc Tanti’s ‘merge model’:

More info on the model here: 
https://machinelearningmastery.com/caption-generation-inject-merge-architectures-encoder-decoder-model/

The model can be split into 3 parts:

Captions:

Character embeddings are found for the characters in the caption (including hashtags, emojis, etc…)
captions are passed into an RNN (with LSTM layer of output size 128)

Images:

Are passed into VGG-16 (minus the last layer)
The result of that is passed as input to a resizing dense layer (128).

Merge:

Resized outputs of images and captions are added together.
Passed through two dense layers, first with ReLu, second with softmax activation.

We train our data in batch sizes of 10, passing in an image with its corresponding caption with all of the characters except the last. We train for 30 epochs. Our loss function uses the whole caption except the first character as the label and the predictions from our model to compute the loss using sparse categorical cross entropy. We use a mask to make sure the padding isn’t used when calculating loss so that we don’t encourage padding in captions. We have a generate caption function that uses our model and a given image to generate a caption character by character. 

## Results:

Success for our project is determined by how much the generated caption consistently resembles a possible human caption. We originally thought about calculating the accuracy through perplexity. However, instagram captions themselves can be quite perplexing so the perplexity we calculate may not be that informative. So the best way to measure success is through human verification. For that reason we did not include an accuracy or testing function. We manually tested by passing in particular images and generating captions and then comparing them to the original captions and just generally to what we think a caption should look like from our experience with instagram. We got our model to generate captions that mostly contain real English words and the occasional hashtag and tags. Our caption never matched the original caption but most of the instagram captions can seem pretty random and not related to the image anyways so it would be impossible to expect the model to learn patterns that just aren’t in the data. Oftentimes there were few random characters in the captions that don’t create full words despite this the captions are legible. 

## Challenges:

Our first challenge was getting our data off instagram, we tried a few different scrapers and APIs, we had to ensure that they were able to get both pictures and captions. A problem we ran into was that instagram shuts you out of their API if you have too many requests in a short amount of time. This meant that we could only run the scraper for around 5 accounts at a time before getting an error which made scraping instagram very time consuming since we had to start and stop the scraper very often. We ended up scraping 100 posts from 58 accounts given out time constraints and the desire to start on preprocessing and the model. 
Another challenge along the way was that when we first ran our model, it kept picking one character seemingly randomly and outputting that character over and over again as our caption. Basically the model wasn’t learning and our gradients were zero. We checked over the model architecture and all the preprocessing which seemed fine. The error turned out to be that when we added the image weights they were overpowering the characters so the model wasn’t able to learn anything, it was getting too much information from the image. To fix the issue we played around with the weight given to the image when added to the captions until we found a good balance that produced reasonable gradients. This change along with other adjustments improved the model to be able to output real words stringed together. Although the grammar of the sentence structure isn’t perfect and some of the words are just random characters at least the captions were readable. This is a challenge resulting from data that doesn’t always have perfect grammar and doesn’t always make sense as a caption for a given image. Although our loss graph does decrease it stops decreasing after around 30 epochs and also reaches a minima that it just goes up and down between. 
 
## Reflection:
Ultimately we are proud of our model for producing actual words and stringing them together even though they don’t necessarily correlate to the image. We ended up not calculating the accuracy as we thought we would when writing our goals since it just didn’t make sense as a metric when we saw our results. We thought we would compare the predicted captions to the actual captions but since they were nowhere close the accuracy would have been useless. Additionally the fact that they weren’t similar to the original caption isn’t that bad as long as they look like any type of caption. As we worked on the project we realized that instagram captions don’t have a formula to them so most caption/image pairs don’t have a reproducible pattern. The most important thing is that the captions are written in an instagram tone meaning users might believe they are captions produced by people which we believe our model is close to achieving. 

If we had to do the project over again, we would pick training data more carefully or take from only certain types of accounts that have captions more directly related to what is in their pictures that way there is more of a correlation between images and captions for the model to pick up patterns from. Also just having more data could drastically improve the model so spending more time on finding a way to scrape from instagram successfully or just taking from a pre made dataset. If we had more time we could play around with the hyperparameters and weights of the images to try and produce better captions. We could also just scrape and train more data. 

Throughout this project we learned more about how different model architectures work and the fact that there isn’t just one way to go about solving a problem. We used an RNN but maybe a transformer model would have been better. We also added the image and caption weights together but maybe we could have passed the images in as a previous state to the LSTM. There are many ways to approach the problem. Additionally, we learned how challenging gathering data can be even from one of the most popular social media sites. We also realized that just getting data isn’t enough, you need to be very specific about the type of data gathered and think about its implications for the model. For example, if we had only gathered data from sport accounts our model might produce very good captions for images relating to different sports but random results for other images. There are trade offs between this and our current model that has information from so many different types of accounts that it didn’t even recognize patterns between image caption pairs. Despite the challenge of actually getting data it is interesting to think about how much data there is in the world around us and all the things (good and bad) it can be used for, this is something we thought about while creating our model. What are other people using these scrapers for? 

