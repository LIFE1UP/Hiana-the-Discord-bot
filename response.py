import pandas as pd
import urllib.request
import re
import random

## Find a Music with Keywords ##
def search(key):
    html = urllib.request.urlopen("https://www.youtube.com/results?search_query=".encode('ascii').decode('utf-8') + key)
    video_ids = re.findall(r"watch\?v=(\S{11})", html.read().decode())

    return "https://www.youtube.com/watch?v=" + video_ids[0]
# def

def spotify_chart(nation):
    html = urllib.request.Request("https://spotifycharts.com/regional/" + nation + "/daily/latest", headers={"User-Agent":"Mozilla/5.0"})
    html = urllib.request.urlopen(html, timeout=10).read().decode()
    music_name =  re.findall(r"<strong>(.{1,100})</strong>", html)
    music_artist = re.findall(r"<span>by (.{1,100})</span>", html)
    key = (lambda x: music_name[x] + " " + music_artist[x])(random.randrange(0, len(music_name) - 1))

    return search(key.replace(" ", "+"))

def spotify_playlist(playlist):
    html = urllib.request.urlopen("https://open.spotify.com/playlist/" + playlist).read().decode()
    music_name = re.findall(r" aria-label=\"track (.{1,100})\" class", html)
    music_artist = re.findall(r"artist/.{24}(.{1,20})\</a></span>", html)
    key = (lambda x: music_name[x] + " " + music_artist[x])(random.randrange(0, len(music_name) - 1))
    
    return search(key.replace(" ", "+"))
# def

## Dictionary ##
fn = {
    # charts
    "korean_music":spotify_chart,
    "us_music":spotify_chart,
    # playlists
    "kawaii_music":spotify_playlist}

url_address = {
    # charts
    "korean_music":"kr",
    "us_music":"us",    
    # playlists
    "kawaii_music":"2oh42HDoUAIwl1LVQ5Z0aR?si=84d6722e141b4194&nd=1"}

## Response ##
def response(tag):
    if not tag in fn:
        return None

    return fn[tag](url_address[tag])
# def

print("response.py ", end="")
