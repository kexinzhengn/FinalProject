using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/* Written by Kexin ZHENG 
* Control the animation of avatar
*/

public class AvatarControl : MonoBehaviour
{
    public bool HAS_ANIMATION{get{return body_rots.Count>0;}}
    List<Vector3> trajectory_3d; // 3d trajectory, one point per frame
    List<Quaternion> body_rots;
    MotionAnimation avatar_animaiton; // should have the same number of frame as the 3d trajectory
    bool is_playing;
 
    float frame_timer;
    string[] joint_names;
    Transform root_joint;

    private static float motion_speed = 50f;
    // Start is called before the first frame update
    void Start()
    {
        is_playing = false;
        frame_timer = 0.0f;
        root_joint = transform.GetChild(2);

        trajectory_3d = new List<Vector3>();
        body_rots = new List<Quaternion>();
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        if(is_playing){
            MotionFrame curr_frame = avatar_animaiton.GetFrame((int)frame_timer);
            MoveJointAtFrame(root_joint, curr_frame);
            transform.position = trajectory_3d[(int)frame_timer];//Vector3.Lerp(transform.position,trajectory_3d[next_frame_idx],frame_timer+1.0f-next_frame_idx);
            transform.rotation = body_rots[(int)frame_timer];
            frame_timer += Time.fixedDeltaTime * motion_speed;
            if(frame_timer >= trajectory_3d.Count-1 || frame_timer >= avatar_animaiton.FRAME_NUM -1 ){
                frame_timer = 0.0f;
                is_playing = false;
            }
        }
    }

    // set the 3D trajectory of the avatar
    public void Set3DTrajectory(Vector3[] traj){
        trajectory_3d.AddRange(traj);
        transform.position = traj[0];
    }

    public void SetBodyRots(Quaternion[] quats){
        body_rots.AddRange(quats);
        transform.localRotation = quats[0];
    }

    public void StartPlaying(){
        if(HAS_ANIMATION)   is_playing = true;
    }

    public void SetAnimation(MotionAnimation am){
        joint_names = am.GetFrame(0).JNAME;
        avatar_animaiton = am.ScaledAnimation(trajectory_3d.Count);
    }

    // Recursivly rotate each joints
    public void MoveJointAtFrame(Transform root, MotionFrame mf){
        root.localRotation = mf.GetJointRotation(root.name);
        for(int i = 0; i < root.childCount;i++){
            MoveJointAtFrame(root.GetChild(i),mf);
        }
    }

    public void ClearAnimation(){
        trajectory_3d.Clear();
        body_rots.Clear();
        avatar_animaiton = null;
    }

}
