# Final System
This folder contains the complete Unity project. Including the final interactive system and a gesture collection application. 

# Unity Version
The current system is built with Unity 2022.3.13f1. Using different version might lead to error during compilation.

# How to Run
1. Clone this project or download ths file from [Google Drive](https://drive.google.com/file/d/1mr1S-H8ROfGjEYZbthwrXQ6j2vrY0Civ/view?usp=sharing)
2. Open it with Unity 2022.3.13f1. Editor
3. The scene for interactive animation system is **Assets/Scenes/Final_System.unity**. The scene for gesture collection application is **Assets/Scenes/Save_Gesture.unity**
4. Click the Run button on the editor. You can now use the system in the 'Game' view.

# Interface
## Animation System
<img src="https://github.com/kexinzhengn/FinalProject/blob/main/FinalSystem/imgs/system_interface.png" width="600">
[Demo Video](https://www.youtube.com/watch?v=KA6DG1JrHv8)
How to create animation with this interface:

1. Click on the 'Draw Line' Button
2. Draw lines follows the motion sketching definition illustrated in the report.
3. After drawing one motion, release the mouse button.
4. Draw the second motion
5. Repeat step 1 to 4 to create combined motion
6. Click on 'Play Animation' button to generate and view the animation
7. You can click on the toggles to switch between different scene views.

## Gesture Collection
<img src="https://github.com/kexinzhengn/FinalProject/blob/main/FinalSystem/imgs/save_interface.png" width="600">
You can save your own gesture using this interface consider different people could have different sketching hobbies. Adding more customized gesture can increse the motion recognition accuracy.

How to save customized gesture
1.  Set the gesture name you want to save in the inspector. Accepted gesture names are 'normal_jump','frontflip', 'backflip', 'run'.<img src="https://github.com/kexinzhengn/FinalProject/blob/main/FinalSystem/imgs/inspector.png" width="600">
2. Click on the 'Draw Line' Button Draw lines follows the motion sketching definition illustrated in the report.
3. Click on the 'Save Gesture' Button. The XML files will be saved to 'Assets/StreamingAssets/CustomGestures'
4. You can view the recognition results by repeat step 1 and 2 then click on the "Recognize Animation' button.
