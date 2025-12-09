  GNU nano 5.4                                              send_notification.py                                                        
import requests

TOPIC = "wolf_detection" # name of topic

def send_text_message(message):
    requests.post(
        f"https://ntfy.sh/{TOPIC}",
        data=message.encode('utf-8'),
        headers={
            "Title": "ALARM", # header of the message
            "Priority": "high", # plays a sound
            "Tags": "warning, wolf" # small icon
        }
    )

send_text_message("WARNING: Wolf detected! Check your sheep immediately!")
