using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

public class MotionLine{
    public Vector3[] DRAWN_PTS{get{return drawn_pts;}}
    public Vector3[] RESAMPLED_DPTS{get{return resampled_drawn_pts;}}
    public string MOTION{get{return motion_name;}}
    public int LENGTH{get{return trajectory_3d.Count;}}
    public Vector3[] TRAJ_3D{get{return trajectory_3d.ToArray();}}

    public Vector3 CLIMBN{get{return climb_norm;}}

    private Vector3[] drawn_pts;
    private Vector3[] resampled_drawn_pts;
    private List<Vector3> trajectory_3d;
    private string motion_name;
    private Vector3 climb_norm;

    public MotionLine(Vector3[] dpt){
        drawn_pts = dpt;
        resampled_drawn_pts = SystemUtils.ResamplePoints(dpt,1.0f);
        trajectory_3d = new List<Vector3>();
        climb_norm = Vector3.zero;
    }


    //// Recognize motion with dolloar recognizer
    public void RecognizeMotion(DollarRecognizer recognizer){
        var resampled_2d = SystemUtils.Vector3To2(resampled_drawn_pts);
        var arc_length = SystemUtils.ComputeArcLength(resampled_drawn_pts);
        var arc_ratio = arc_length / Vector3.Distance(resampled_drawn_pts[0],resampled_drawn_pts[resampled_drawn_pts.Length-1]);
        if(arc_ratio<1.2){ //  straight line
            motion_name = "walk";
            Debug.Log(motion_name);
            return;
        }
        var res = recognizer.Recognize(resampled_2d);
        if(res.Match.Name.Contains("run")) motion_name = "run";
        else if(res.Match.Name.Contains("normal_jump")) motion_name = "normal_jump";
        else if(res.Match.Name.Contains("frontflip")) motion_name = "frontflip";
        else if(res.Match.Name.Contains("backflip")) motion_name = "backflip";
        if(motion_name.Contains("run")){
            resampled_drawn_pts = SystemUtils.ResamplePoints(drawn_pts,3.0f); // faster
        }
    }

    public void Generate3DLine(GameObject target_scene){
        var cam = Camera.main;
        /// For non-planar motion i.e. JUMP
        if(motion_name.Contains("jump")||motion_name.Contains("flip")){
            // only raycast the start and end point
            Vector3 spt, snorm, ept, enorm;
            spt = snorm = ept = enorm = Vector3.zero;
            GetHitPointAndNormal(resampled_drawn_pts[0],cam,target_scene,ref spt,ref snorm);
            GetHitPointAndNormal(resampled_drawn_pts[resampled_drawn_pts.Length-1],cam,target_scene,ref ept,ref enorm);
            trajectory_3d.Add(spt);
            for(int i = 1; i < resampled_drawn_pts.Length;i++){
                var ray = cam.ScreenPointToRay(resampled_drawn_pts[i]);
                Vector3 curve_pt = GetProjectedPointOnPlane(ray,spt,ept,spt+Vector3.up);
                trajectory_3d.Add(curve_pt);
            }
            trajectory_3d.Add(ept);
            return;
        }

        //// For motion lies on a plane
        var new_pts = SmoothGesturePoints(resampled_drawn_pts);
        for (int i = 0; i < new_pts.Length;i++){// loop points
            var ray = cam.ScreenPointToRay(new_pts[i]);
            RaycastHit Hit;
            if (Physics.Raycast(ray, out Hit))
            {
                for(int j = 0; j<target_scene.transform.childCount;j++){// loop scene objects
                    var target = target_scene.transform.GetChild(j);
                    if (Hit.transform == target.transform){
                        var hit_point = Hit.point;
                        trajectory_3d.Add(hit_point);
                        break;
                    }
                }
            }
        }
        // modify motion for climb
        if(Mathf.Abs(trajectory_3d[0].y-trajectory_3d[trajectory_3d.Count-1].y)>0.1f){
            motion_name = "climb";
            Debug.Log(motion_name);
            Vector3 cpt, cnorm;
            cpt = cnorm = Vector3.zero;
            GetHitPointAndNormal(resampled_drawn_pts[0],cam,target_scene,ref cpt,ref cnorm);
            climb_norm = cnorm;
            for(int i = 0; i < trajectory_3d.Count;i++){
                trajectory_3d[i] += cnorm * 0.15f;
                trajectory_3d[i] -= new Vector3(0.0f,0.2f,0.0f);
            }
        }
    }

    //// smooth gesture point to trajectory
    Vector3[] SmoothGesturePoints(Vector3[] gesture_pts){
        var new_pts = SystemUtils.MidpointSmoothing(gesture_pts);
        new_pts = SystemUtils.DouglasPeuckerReduction(SystemUtils.Vector3To2(new_pts),10.0f).ToArray();
        if(new_pts.Length>2) new_pts = SystemUtils.MidpointSmoothing(new_pts);
        new_pts = SystemUtils.ResamplePoints(new_pts,1.0f);
        return new_pts;
    }

    void GetHitPointAndNormal(Vector3 tpt, Camera cam, GameObject target_scene, ref Vector3 hpt, ref Vector3 hnorm){
        var ray = cam.ScreenPointToRay(tpt);
        RaycastHit Hit;
        if(Physics.Raycast(ray,out Hit)){
            for(int i = 0; i < target_scene.transform.childCount;i++){
                var target = target_scene.transform.GetChild(i);
                if(Hit.transform == target.transform){
                    hpt = Hit.point;
                    hnorm = Hit.normal;
                }
            }
        }
    }

    Vector3 GetProjectedPointOnPlane(Ray ray, Vector3 p1, Vector3 p2, Vector3 p3){
        Vector3 p12 = p1 - p2;
        Vector3 p32 = p3 - p2;
        Vector3 pnorm = Vector3.Cross(p12,p32).normalized;
        float dist = Vector3.Dot(p1 - ray.origin,pnorm)/Vector3.Dot(ray.direction,pnorm);
        Vector3 result = ray.GetPoint(dist);
        return result;
    }

    //// Visualization of 3D trajectory
    void VisualizeWithSphere(Vector3 pos){
        var sp = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        sp.transform.localScale = new Vector3(0.05f,0.05f,0.05f);
        sp.transform.position = pos;//tt + new Vector3(0.0f,0.0f,0.005f);
    }

    public Vector3 GetHighestPoint(ref int idx){
        var hpt = trajectory_3d[0];
        for(int i = 0; i < trajectory_3d.Count;i++){
            if(trajectory_3d[i].y>hpt.y) {
                idx = i;
                hpt = trajectory_3d[i];
            }
        }
        return hpt;
    }

    public Vector3[] SamplePointsFromOffsetAndHeight(int aesnum,int desnum){
        var target_num = aesnum + desnum;
        var results = new Vector3[target_num];
        int hidx =0;
        Vector3 hpt = GetHighestPoint(ref hidx);
        //aescending
        int[] sample_numa = new int[aesnum];
        float sum_portion = 0.0f;
        sample_numa[0] = 0;
        for(int i = 1; i < aesnum;i++){
            sum_portion += (float)(i*i);
            sample_numa[i] = sample_numa[i-1] + (aesnum-i)*(aesnum-i);
        }
        sample_numa[aesnum-1] = (int)sum_portion;
        float sub_sample = (float)((hpt.y-TRAJ_3D[0].y)/sum_portion);
        for(int i = 0; i<aesnum;i++){
            float curr_y = TRAJ_3D[0].y + sub_sample * (float)sample_numa[i];
            Vector3 lerpvec = Vector3.Lerp(TRAJ_3D[0], TRAJ_3D[LENGTH-1],(float)(i)/(float)(target_num-1.0f));
            results[i] = new Vector3(lerpvec.x,curr_y,lerpvec.z);
        }
        //des
        int[] sample_numd = new int[desnum];
        sum_portion = 0.0f;
        sample_numd[0] = 0;
        for(int i = 1; i < desnum;i++){
            sum_portion += (float)(i*i);
            sample_numd[i] = sample_numd[i-1] + (i)*(i);
        }
        sample_numd[desnum-1] = (int)sum_portion;
        sub_sample = (hpt.y-TRAJ_3D[LENGTH-1].y)/sum_portion;
        for(int i = 0; i<desnum;i++){
            float curr_y = hpt.y - sub_sample * (float)sample_numd[i];
            Vector3 lerpvec = Vector3.Lerp(TRAJ_3D[0], TRAJ_3D[LENGTH-1], (float)(i + aesnum)/(float)(target_num-1.0f));
            results[i+aesnum] = new Vector3(lerpvec.x,curr_y,lerpvec.z);
        }
        return results;
    }

    public Vector3[] SamplePointsFromHeight(int target_num, int sid, int eid, bool ascend){
        var subpts = SystemUtils.SubArray(trajectory_3d.ToArray(),sid,eid-sid+1);
        var rsubpts = SystemUtils.ResamplePoints(subpts,0.005f);
        var results = new Vector3[target_num];
        float sum_portion = 0.0f;
        int[] sample_num = new int[target_num];
        sample_num[0] = 0;
        if(ascend){
            var hpt = subpts[subpts.Length-1];
            for(int i = 1; i < target_num;i++){
                sum_portion += (float)(i*i);
                sample_num[i] = sample_num[i-1] + (target_num-i)*(target_num-i);
            }
            float sub_sample = (float)((hpt.y-subpts[0].y)/sum_portion);
            Debug.Log(sub_sample + "sbsamplel");
            for(int i = 0; i<target_num;i++){
                float curr_y = subpts[0].y + sub_sample * (float)sample_num[i];
                results[i] = SystemUtils.FindPointWithHeight(rsubpts,curr_y);
            }
        }else{
            for(int i = 1; i<target_num;i++){
                sum_portion += (float)(i*i);
                sample_num[i] = sample_num[i-1] + (i*i);
            }
            var hpt = subpts[0];
            var ept = subpts[subpts.Length-1];
            float sub_sample = (float)(hpt.y-ept.y)/sum_portion;
            for(int i = 0; i<target_num;i++){
                float curr_y = hpt.y - sub_sample * (float)sample_num[i];
                results[i] = SystemUtils.FindPointWithHeight(rsubpts,curr_y);
            }

        }
        return results;
    }


}