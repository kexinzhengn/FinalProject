using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

/*
* Written by ZHENG Kexin
* The main controller script that control the overall interaction and viewing options of the system
*/
public class MainController : MonoBehaviour
{
    // UIs
    [SerializeField] 
    public Button draw_btn;
    public Button createline_btn;
    public Toggle scene_toggle;
    public Toggle semantic_toggle;
    public Toggle sphere_toggle;

    // Gameobjects
    public GameObject avatar_model; // animation avatar model
    public GameObject target_scene; // gameobject contains scene objects
    public GameObject real_scene;
    public GameObject semantic_scene;

    // Public variables
    public string gesture_path; // folder that contains the gesture data
    public float cammove_scale;
    public float camrot_scale;
    private DrawLine line_control; // draw line helper script
    private DollarRecognizer recognizer; // dolloar gesture recognizor
    private AnimationModifier animation_modifier; // Load and modify motions
    private AvatarControl avatar_control; // Control the playing of animation
    // Start is called before the first frame update
    void Start()
    {
        // UI settings
        draw_btn.onClick.AddListener(ActivateDraw);
        createline_btn.onClick.AddListener(CreateMotionLines);
        scene_toggle.onValueChanged.AddListener(delegate {
            SceneValueChanged(scene_toggle);
        });
        semantic_toggle.onValueChanged.AddListener(delegate{
            SemanticValueChanged(semantic_toggle);
        });
        sphere_toggle.onValueChanged.AddListener(delegate {
            SphereValueChanged(sphere_toggle);
        });
        // Gameobject settings
        real_scene.SetActive(false);
        semantic_scene.SetActive(false);
        target_scene.SetActive(true);
        // draw helper
        line_control = transform.GetComponent<DrawLine>();
        // recognizer
        recognizer = new DollarRecognizer(gesture_path);// initialize gesture recognizer
        // animation modifier
        animation_modifier = new AnimationModifier();
        // avatar control
        avatar_control = avatar_model.GetComponent<AvatarControl>();
    }

    // Update is called once per frame
    // Camera Controls
    void Update()
    {
        /// 3 fixed scene view for easy navigation for demonstration
        if(Input.GetKey("1")){
            Camera.main.transform.position = new Vector3(-1.89f,1.4f,-4.02f);
            Camera.main.transform.rotation = Quaternion.Euler(17.988f, 184.521f, 34.341f);
        }else if(Input.GetKey("2")){
            Camera.main.transform.position = new Vector3(-0.86f,2.57f,-4.06f);
            Camera.main.transform.rotation = Quaternion.Euler(24.129f, 275.503f, -4.548f);
        }else if(Input.GetKey("3")){
            Camera.main.transform.position = new Vector3(-5.96f,2.83f,-5.27f);
            Camera.main.transform.rotation = Quaternion.Euler(41.841f, 364.369f, 28.873f);
        }
        // Scroll to move camera forward and backward
        if(Input.mouseScrollDelta.y!=0){
            Camera.main.transform.position -= Camera.main.transform.forward * Input.mouseScrollDelta.y * cammove_scale;
        }
        // Press the scroll on mouse to move the camera horizontally and vertically
        if(Input.GetMouseButton(2)){
            Camera.main.transform.position -= Camera.main.transform.right * Input.GetAxis("Mouse X") * cammove_scale;
            Camera.main.transform.position -= Camera.main.transform.up *  Input.GetAxis("Mouse Y") * cammove_scale;
        }
        // Press left button to rotate the camera
        if(!line_control.IS_DRAWING){
            if(Input.GetMouseButton(0)) {
                Camera.main.transform.Rotate(new Vector3(-Input.GetAxis("Mouse Y") * camrot_scale, -Input.GetAxis("Mouse X")*camrot_scale, 0), Space.World);
		    }
        }
    }

    void CreateMotionLines(){
        line_control.stopDraw();
        if(!avatar_control.HAS_ANIMATION){
            var mouse_pts = line_control.MOUSE_PTS;
            var seg_idxs = line_control.SEG_IDXS;
            List<MotionLine> motion_lines = new List<MotionLine>();
            for(int i = 0; i < seg_idxs.Length;i++){
                int prev_idx = (i==0)? 0 : seg_idxs[i-1];
                var curr_pts = SystemUtils.SubArray(mouse_pts,prev_idx,seg_idxs[i]-prev_idx+1);
                MotionLine curr_line = new MotionLine(curr_pts);
                curr_line.RecognizeMotion(recognizer);
                curr_line.Generate3DLine(target_scene);
                motion_lines.Add(curr_line);
            }
            animation_modifier.CompositeAnimation(motion_lines.ToArray());
            // set animation data to avatar object
            avatar_control.Set3DTrajectory(animation_modifier.TRAJECTORY);
            avatar_control.SetAnimation(animation_modifier.ANIMATION);
            avatar_control.SetBodyRots(animation_modifier.BODY_ROTS);
        }
        avatar_control.StartPlaying();
    }

    // Activate Drawing 
    void ActivateDraw(){
        line_control.activateDraw();
        avatar_control.ClearAnimation();
        animation_modifier.ClearAnimation();
        DestroySphereVisualization();
    }

    // Control visibility of real scene viewer
    void SceneValueChanged(Toggle change)
    {
        real_scene.SetActive(scene_toggle.isOn);
        target_scene.SetActive(!scene_toggle.isOn);
        line_control.ChangeLineView(!scene_toggle.isOn);
    }

    // Control visibility of semantic scene view
    void SemanticValueChanged(Toggle change){
        semantic_scene.SetActive(semantic_toggle.isOn);
        target_scene.SetActive(!semantic_toggle.isOn);
        line_control.ChangeLineView(!semantic_toggle.isOn);
    }

    // Control visibility of moving trajectory
    void SphereValueChanged(Toggle change){
        var spheres = GameObject.FindGameObjectsWithTag("vizsphere");
        foreach(var sp in spheres){
            sp.GetComponent<MeshRenderer>().enabled = sphere_toggle.isOn;
        }
    }

    // Remove 3D trajectory visualizer
    void DestroySphereVisualization(){
        var spheres = GameObject.FindGameObjectsWithTag("vizsphere");
        foreach(var sp in spheres){
            Destroy(sp);
        }
    }
}
