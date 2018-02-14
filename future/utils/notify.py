from slackclient import SlackClient
import os
import json
slack_token = "xoxb-312747468528-xmPcgWMg8MhZeMRFDWaV8eYC"
sc = SlackClient(slack_token)
def notify(msg):
    a = sc.api_call(
      "chat.postMessage",
      channel="#stock",
      text=msg
    )
