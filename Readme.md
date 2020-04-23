enroll inputs :

- name of user
- image of user

# inscription :
- For the main inscription, this is an example of the curl request to send :
```
curl -X POST -F cin_file=@ihab.jpg 'http://127.0.0.1:5000/api/enroll?username=ihab'
```
the `cin_file` is the path to the image of the cin of the user , and the username is the username with which he'll be saved in the database

All cases (missing username, missing cin image, duplicate username, empty username, or empty cin file) have been taken care of, and the server would return an answer for all cases.


# AUTHENTIFICATION

- For the main authentication, this is an example of the curl request to send :

```
curl -X POST -F auth_video=@ihab.jpg 'http://127.0.0.1:5000/api/authentificate?username=ihab'
```

the `auth_video` is the path to the video of the user , and the username is the username with which he is saved in the database

All cases (missing username, missing auth video, empty username, or empty video file) have been taken care of, and the server would return an answer for all cases.
