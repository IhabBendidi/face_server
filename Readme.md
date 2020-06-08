
# inscription :
- For the main inscription, this is an example of the curl request to send in terminal:
```
curl -X POST -F cin_file=@brahim.jpeg 'http://127.0.0.1:5000/api/enroll?username=brahim'
```
the `cin_file` is the path to the image of the cin of the user , and the username is the username with which he'll be saved in the database

All cases (missing username, missing cin image, duplicate username, empty username, or empty cin file) have been taken care of, and the server would return an answer for all cases.

On sucess, it returns a json in the form of :

```
{"trained":"success","EnrollStatus":"APPROVED","DecisionReason":"ENROLLED_AS_USER"}
```

On failure, it returns a json in the form of the following, with `DecisionReason` being the reason of the specific failure :

```
{"trained":"failed","EnrollStatus":"REJECTED","DecisionReason":"THE_FAILURE_MESSAGE"}
```


# AUTHENTIFICATION

- For the main authentication, this is an example of the curl request to send in terminal :

The liveness model currently has a weak f1 score, so it is desactivated in the server for now until we can raise further its recall metric. the `auth_video` takes for now only an image of the user

```
curl -X POST -F auth_video=@brahim.mp4 'http://127.0.0.1:5000/api/authentificate?username=brahim'
```

the `auth_video` is the path to the video of the user , and the username is the username with which he is saved in the database

All cases (missing username, missing auth video, empty username, or empty video file) have been taken care of, and the server would return an answer for all cases.


On sucess, it returns a json in the form of :

```
{"score":<the_confidence_score>,"Decision":"APPROVED","DecisionReason":"AUTHENTIFICATION_SUCCESS"}
```

On failure, it returns a json in the form of the following, with `DecisionReason` being the reason of the specific failure :

```
{"score":None,"Decision":"REJECTED","DecisionReason":"THE_FAILURE_MESSAGE"}
```



# Deleting all users in server :
```
curl -X POST 'http://127.0.0.1:5000/api/reset'
```


```
{"code":200,"msg":"DONE"}

{"code":500,"msg":"ERROR_DETECTED : <Explanation of error>"}
```
