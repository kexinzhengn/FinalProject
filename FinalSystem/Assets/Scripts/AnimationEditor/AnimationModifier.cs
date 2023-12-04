using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/*
* Written by ZHENG Kexin
* Modify animation preset to align with 3D trajectory
*/
public class AnimationModifier{
    public MotionAnimation ANIMATION{get{return curr_animation;}}
    public Vector3[] TRAJECTORY{get{return curr_trajectory.ToArray();}}
    public Quaternion[] BODY_ROTS{get{return curr_quats.ToArray();}}
    MotionAnimation curr_animation;
    List<Vector3> curr_trajectory;
    List<Quaternion> curr_quats;
    Dictionary<string, MotionAnimation> motion_dict;
    float max_jump_height = 2.5f;
    int max_jump_frame = 25;

    public AnimationModifier(){
        // Load Motion from files
        motion_dict = new Dictionary<string, MotionAnimation>();
        var walk_anim = FBXAnimationLoader.LoadFBXAnimation("walk","AnimationClip/Walking");
        var jump_anim = FBXAnimationLoader.LoadFBXAnimation("normal_jump","AnimationClip/Jump");
        var run_anim = FBXAnimationLoader.LoadFBXAnimation("run","AnimationClip/Run");
        var climb_anim = FBXAnimationLoader.LoadFBXAnimation("climb", "AnimationClip/Climbing");
        var ff_anim = FBXAnimationLoader.LoadFBXAnimation("frontflip","AnimationClip/frontflip");
        var bf_anim = FBXAnimationLoader.LoadFBXAnimation("backflip","AnimationClip/backflip");
        motion_dict.Add("walk",walk_anim);
        motion_dict.Add("normal_jump",jump_anim);
        motion_dict.Add("run",run_anim);
        motion_dict.Add("climb",climb_anim);
        motion_dict.Add("frontflip",ff_anim);
        motion_dict.Add("backflip",bf_anim);
        // trajectory
        curr_trajectory = new List<Vector3>();
        // rotation
        curr_quats = new List<Quaternion>();
    }

    // combine the 3D trajectory and animation of all points
    public void CompositeAnimation(MotionLine[] motionLines){
        curr_trajectory.Clear();
        curr_animation = new MotionAnimation("composite");
        for(int i = 0; i < motionLines.Length;i++){
            var curr_line = motionLines[i];
            Debug.Log(curr_line.MOTION + "final seg");
            int asf, dsf;
            asf = dsf = 0;
            int pt_in_traj = curr_trajectory.Count;
            if(curr_line.MOTION.Contains("jump")||curr_line.MOTION.Contains("flip")) ModifyJumpTrajectory(curr_line,ref asf, ref dsf);
            else curr_trajectory.AddRange(curr_line.TRAJ_3D);
            var new_anim = ModifyMotion(curr_line,asf,dsf);
            curr_animation.AddMotionFrames(new_anim.FRAMES); // edit
            ComputeBodyRotation(pt_in_traj,curr_line);
        }
       VisualizeTrajectory();
    }

    public void ModifyJumpTrajectory(MotionLine curr_line, ref int aseFrame, ref int dseFrame){
        // anticipation
        if(curr_line.MOTION == "normal_jump"){
            for(int j=0; j < 23;j++){// anticipation
                    curr_trajectory.Add(curr_line.TRAJ_3D[0]);
            }
        }
        else if(curr_line.MOTION =="backflip"){
            for(int j = 0; j < 34;j++){
                curr_trajectory.Add(curr_line.TRAJ_3D[0]);
            }
        }
        // aescending and descending
        int top_idx = 0;
        Vector3 toppt = curr_line.GetHighestPoint(ref top_idx);
        float ascendDist = toppt.y - curr_line.TRAJ_3D[0].y;
        float desDist = toppt.y - curr_line.TRAJ_3D[curr_line.TRAJ_3D.Length-1].y;
        // frame number proportional to jump height
        int aes_frame = (int)(Mathf.Sqrt(ascendDist/max_jump_height) * (float)max_jump_frame);
        int des_frame = (int)(Mathf.Sqrt(desDist/max_jump_height) * (float)max_jump_frame);

        if(curr_line.MOTION=="normal_jump"){
            curr_trajectory.AddRange(curr_line.SamplePointsFromHeight(aes_frame,0,top_idx,true));
            curr_trajectory.AddRange(curr_line.SamplePointsFromHeight(des_frame,top_idx,curr_line.LENGTH-1,false));
        }else if(curr_line.MOTION.Contains("flip")){
            curr_trajectory.AddRange(curr_line.SamplePointsFromOffsetAndHeight(aes_frame,des_frame));
        }
        // landing
        if(curr_line.MOTION=="normal_jump"){ // landing
            for(int j = 0; j < 35;j++){
                curr_trajectory.Add(curr_line.TRAJ_3D[curr_line.TRAJ_3D.Length-1]);
            }
        }else if(curr_line.MOTION=="backflip"){
            for(int j = 0; j <63;j++){
                curr_trajectory.Add(curr_line.TRAJ_3D[curr_line.TRAJ_3D.Length-1]);
            }
        }
        aseFrame = aes_frame;
        dseFrame = des_frame;
    }

    public MotionAnimation ModifyMotion(MotionLine motion_line, int ase_frame, int dse_frame){
        // one frame at a point
        var new_motion = new MotionAnimation(motion_line.MOTION+"_modified");
        if(motion_line.MOTION.Contains("normal_jump")) {// jump
            new_motion = motion_dict[motion_line.MOTION].ScaleJumpAnimation(ase_frame, dse_frame, 23,31, 42);
        }else if(motion_line.MOTION=="frontflip"){
            new_motion = motion_dict[motion_line.MOTION].ScaleJumpAnimation(ase_frame, dse_frame, 0,16, 32);
        }else if(motion_line.MOTION=="backflip"){
            new_motion = motion_dict[motion_line.MOTION].ScaleJumpAnimation(ase_frame, dse_frame, 34,45,59);
            Debug.Log(new_motion.FRAME_NUM+"asdadad"+curr_trajectory.Count);
        }
        else {// walk or running or climb
            new_motion = motion_dict[motion_line.MOTION].RepeatMotions(motion_line.LENGTH);
        }
        
        return new_motion;
    }

    void ComputeBodyRotation(int curr_line_start, MotionLine motion_line){
        for(int i = curr_line_start; i < curr_trajectory.Count;i++){
            // get moving direction
            var move_dir = Vector3.right;
            if(i==0) {
                move_dir = curr_trajectory[i+1] - curr_trajectory[i];
            }
            else {
                move_dir = curr_trajectory[i] - curr_trajectory[i-1];
            }
            if(motion_line.MOTION.Contains("jump")||motion_line.MOTION.Contains("frontflip")){
                move_dir = curr_trajectory[curr_trajectory.Count-1] - curr_trajectory[curr_line_start];
            }else if(motion_line.MOTION=="backflip"){
                move_dir = curr_trajectory[curr_line_start] - curr_trajectory[curr_trajectory.Count-1];
            }else if(motion_line.MOTION.Contains("climb")){
                move_dir = motion_line.CLIMBN;
            }
            move_dir.y = 0.0f; // do not change the up direction value
            var avatar_quat = Quaternion.LookRotation(-move_dir.normalized);
            curr_quats.Add(avatar_quat);
        }
    }

    public void ClearAnimation(){
        curr_animation = null;
        curr_trajectory.Clear();
        curr_quats.Clear();
    }

    void VisualizeTrajectory(){
        for(int i = 0; i<curr_trajectory.Count;i++){
            VisualizeWithSphere(curr_trajectory[i]);
        }
    }

    void VisualizeWithSphere(Vector3 pos){
        var sp = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        sp.tag = "vizsphere";
        sp.transform.parent = GameObject.FindGameObjectsWithTag("Player")[0].transform;
        sp.transform.localScale = new Vector3(0.05f,0.05f,0.05f);
        sp.transform.position = pos;//tt + new Vector3(0.0f,0.0f,0.005f);
        sp.GetComponent<Renderer>().material.SetColor("_Color", Color.grey);
    }

}