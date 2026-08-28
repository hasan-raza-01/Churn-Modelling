import winsound, time, math, os
from plyer import notification
from churn_modelling.configuration import ModelTrainerConfig

duration = 60*60*24 # 24 hours
stop_time = math.ceil(time.time()+(duration))

model_trainer_config = ModelTrainerConfig()
scores_file_path = model_trainer_config.SCORES_FILE_PATH

# check file availablility
while True: 
    if os.path.exists(scores_file_path): 
        print(end="\n\n\n")
        print("X"*100)
        print("😭"*10, "Finally Model Training completed", "😭"*10)
        print("X"*100)
        break
    print("yet training not completed................")
    time.sleep(60*5) # 5 min

notification.notify(
    title='Alert!',
    message='Model Training have been completed',
    app_name='Python Application',
    timeout=duration # Duration in seconds the notification stays on screen
)
while True:
    if time.time()>=stop_time: 
        break
    winsound.Beep(440, 100)
    time.sleep(0.1)

