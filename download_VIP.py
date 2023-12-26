from pytube import YouTube
import tensorflow as tf
import pandas as pd
from IPython.display import YouTubeVideo
from google.cloud import storage, exceptions
import re
import shlex, subprocess
import pandas as pd
import os


def returnNumberOfKeyFrames(link: str):
    """Given a youtube link, will find video length then will return the number of keyframes--1 per every 15 seconds"""
    return (YouTube(link).length) // 15


def Download(link: str, id: str, outpath: str = "./"):
    youtubeObject = YouTube(link)
    youtubeObject = youtubeObject.streams.get_highest_resolution()
    try:
        youtubeObject.download(output_path=outpath, filename=f"{id}.mp4")
    except:
        print("An error has occurred")
    print("Download is completed successfully")


def getYoutubeID(code: str):
    cmd = f"curl http://data.yt8m.org/2/j/i/{code[0:2]}/{code}.js"
    args = shlex.split(cmd)
    process = subprocess.Popen(
        args, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    s = stdout.decode("utf-8")
    if "Error" in s:
        return "Error"
    else:
        return re.findall('"([^"]*)"', s)[1]


def getVidsandCategories(start: int, end: int, record="./train00.tfrecord"):
    """Choose which training file to process
    Returns list of videos
    """
    vid_ids = []
    labels = []

    for example in tf.compat.v1.python_io.tf_record_iterator(record):
        seq_example = tf.train.Example.FromString(example)
        vid_ids.append(
            seq_example.features.feature["id"]
            .bytes_list.value[0]
            .decode(encoding="UTF-8")
        )
        labels.append(seq_example.features.feature["labels"].int64_list.value)
    print(len(vid_ids))

    vocabulary = pd.read_csv("/data4/ersp2022/videos/vocabulary.csv")
    vocabulary.set_index("Index", inplace=True)

    ids_labels = []
    """Extract id's here"""
    for i in range(len(vid_ids)):
        categories = set()
        ytb_id = getYoutubeID(vid_ids[i])
        if ytb_id != "Error":
            for label in labels[i]:
                category1 = vocabulary["Vertical1"][label]
                category2 = vocabulary["Vertical2"][label]
                category3 = vocabulary["Vertical3"][label]
                if pd.notna(category1):
                    categories.add(category1)
                if pd.notna(category2):
                    categories.add(category2)
                if pd.notna(category3):
                    categories.add(category3)
            ids_labels.append([ytb_id, list(categories)])
    return ids_labels


def removeCategories(videosAndCategories: list):
    """
    Takes in a list of video ids (in YouTube format) as well as categories it fits into. Returns a pruned list of only good ones
    """
    finalList = []
    # goodCategories = ["Autos & Vehicles", "Food & Drink", "Hobbies & Leisure", "Travel"]
    goodCategories = [
        "Hobbies & Leisure",
        "Pets & Animals",
        "Food & Drink",
        "Travel",
        "People and Society",
        "Books and Literature",
        "Outdoor Recreation",
        "Jobs & Education",
        "Sports",
        "Law & Government",
        "Home & Garden",
        "Real Estate",
    ]
    badCategories = [
        "Games",
        "Arts & Entertainment",
        "Computers & Electronics",
        "Shopping",
        "Beauty & Fitness",
        "Internet & Telecom",
        "Hairstyle",
        "News",
        "Science",
    ]
    for video in videosAndCategories:
        categories = video[1]
        posNegCategoriesSum = 0
        wasBadCategory = False
        if categories == ["Sports"]:
            continue
        if "Autos & Vehicles" in categories:
            continue
        for category in categories:
            if category in goodCategories:
                posNegCategoriesSum += 1
            if category in badCategories:
                posNegCategoriesSum -= 1
                wasBadCategory = True
        if posNegCategoriesSum > 0 or (posNegCategoriesSum == 0 and not wasBadCategory):
            finalList.append(video)
    return finalList


def downloadVideos(
    start: int, end: int, downloadPath: str, record="./train00.tfrecord"
):
    vids = getVidsandCategories(start=start, end=end, record=record)
    print(vids)
    finalVideos = removeCategories(vids)
    print(finalVideos)
    for video in finalVideos:
        if not os.path.exists(f"/data4/ersp2022/videos/VIPvideos/{video[0]}.mp4"):
            try:
                Download(
                    f"https://www.youtube.com/watch?v={video[0]}",
                    video[0],
                    outpath=downloadPath,
                )
            except:
                print("Error downloading video")


def main():
    # files = ["trainad.tfrecord","trainDC.tfrecord","trainOG.tfrecord","trainOK.tfrecord","trainpj.tfrecord","trainXY.tfrecord",]
    for file in os.listdir("/data4/ersp2022/videos/finalTrain"):
        downloadVideos(
            start=0,
            end=0,
            downloadPath="/data4/ersp2022/videos/VIPvideos/",
            record=f"/data4/ersp2022/videos/finalTrain/{file}",
        )
    # # getVidsandCategories(1000, 1005, record = "/data4/ersp2022/videos//trainWB.tfrecord")
    # print(returnNumberOfKeyFrames("https://www.youtube.com/watch?v=AugoGXy1dZo"))
    # print(os.listdir("/data4/ersp2022/videos/trialvideos"))
    print(returnNumberOfKeyFrames("https://www.youtube.com/watch?v=JXQr8MoAuXI"))


if __name__ == "__main__":
    main()
