# EMG Decoding Demo - Flappy Bird controlled by EMG

## Setup and installation
### Hardware
- Personal Computer
- Raspberry Pi Zero 2 W
- High-Precision AD HAT (already connected to the rpi)
- EMG channels with bio-amp (connected to the AD HAT)

### Software
#### Raspberry Configuration

1. Download nmap https://nmap.org/download and Raspberry Pi Imager
2. Do this step ONLY if you need to reconfigure the Wi-Fi --> will delete all files in the rpi : Reboot the micro SD using the Imager
    1. Model : Raspberry Pi Zero 2 W
    2. OS : Legacy, 32-bit
    3. Stockage : micro SD
    4. Settings : 
        1. host : raspberrypi.local
        2. user : npulse, password : npulse2025
        3. Wi-Fi : [your_hotspot], password : [your_password]
3. Connect to a private wifi (hotspot)
4. Type the following command on the terminal
    
    ```jsx
    ipconfig
    ```
    
5. Take the IPv4 adress. Replace the stars below with the 3 first blocks of numbers of this adress and type it on the terminal. It will search all IP adresses: search for the Raspberry one.
    
    ```jsx
    nmap -sn ***.***.***.0/24
    ```
    
6. Type the following command on PowerShell to begin the ssh connexion. Type yes to continue connecting (if requested), and enter the password (npulse2025).
    
    ```jsx
    ssh npulse@***.***.***.***
    ```
    
7. If you have an error related to ssh, you can try this
    
    ```jsx
    ssh-keygen -R <host>
    ```
    
8. You can now navigate inside the Raspberry. To close the connection, type Ctrl+D
9. To navigate between folders. You can use the following commands :
    1. “cd /path”, “ls” and “pwd” : navigate (move from one folder to another; check what is in the current folder; check the current location)
    2.  Make a new folder
        
        ```jsx
        mkdir newproject
        ```
        
    3. Copy file from computer to Raspberry
        
        ```jsx
        scp C:\path\to\your\[file.py](http://file.py/) npulse@***.***.***.***:/home/pi/
        ```
        
    4. Run python script
        
        ```jsx
        python3 script.py
        ```
        
#### File/folder installation 
- flappy_demo : on your personal computer. Run stream.py to launch the flappy game. Run test_socket.py to check the Socket (UDP) connexion.
- rpi_UDP : on the raspberry. Run new_please.py to begin the Socket (UDP) connexion.
