# Import the following modules
import requests
import json
import os


# Function to send Push Notification


def pushbullet_notification(title, body):
    TOKEN = 'o.lrmGt70Q3N2gAB6Q3o92Z4l5neI9JSEY'  # Pass your Access Token here
    # Make a dictionary that includes, title and body
    msg = {"type": "note", "title": title, "body": body}
    # Sent a posts request
    resp = requests.post('https://api.pushbullet.com/v2/pushes',
                         data=json.dumps(msg),
                         headers={'Authorization': 'Bearer ' + TOKEN,
                                  'Content-Type': 'application/json'})
    if resp.status_code != 200:  # Check if fort message send with the help of status code
        print("Error sending notification...")
        # raise Exception('Error', resp.status_code)


def pushbullet_image(image_path, image_type='png'):
    TOKEN = 'o.lrmGt70Q3N2gAB6Q3o92Z4l5neI9JSEY'  # Pass your Access Token here
    # Make a dictionary that includes, title and body
    # msg = {"type": "file", "file_type": "image/jpeg", "file_name": "test.png", "file_url": str(encoded_string)}
    resp = requests.post('https://api.pushbullet.com/v2/upload-request', data=json.dumps({'type': 'file', 'file_name': image_path.split(os.sep)[-1]}),
                         headers={'Authorization': 'Bearer ' + TOKEN, 'Content-Type': 'application/json'})
    if resp.status_code != 200:
        raise Exception('failed to request upload')
    r = resp.json()
    resp = requests.post(r['upload_url'], data=r['data'], files={'file': open(image_path, 'rb')})
    if resp.status_code != 204:
        raise Exception('failed to upload file')
    msg = {"type": "file", "file_type": f"image/{image_type}", "file_name": image_path.split(os.sep)[-1], "file_url": r['file_url']}
    resp = requests.post('https://api.pushbullet.com/v2/pushes',
                         data=json.dumps(msg),
                         headers={'Authorization': 'Bearer ' + TOKEN,
                                  'Content-Type': 'application/json'})
    if resp.status_code != 200:  # Check if fort message send with the help of status code
        print("Error sending notification...")


if __name__ == '__main__':
    pushbullet_notification(f"This is a test", f"Teeest")
