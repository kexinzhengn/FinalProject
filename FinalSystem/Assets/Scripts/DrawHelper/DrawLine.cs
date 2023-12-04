using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/* Written by Kexin ZHENG 
* Stores drawn line information 
*/
public class DrawLine : MonoBehaviour
{
    // retrieve data
    public Vector3[] MOUSE_PTS{get{return mousePoints.ToArray();}}
    public int[] SEG_IDXS{get{return stopIndex.ToArray();}}
    public bool IS_DRAWING{get{return isDrawing;}}

    private List<Vector3> mousePoints; // get screen position
    private List<int> stopIndex; // index of mousePoints with the drawing stop(one segment)

    // prefabs
    public GameObject lineDrawerPrefab;
    // basic setting
    private LineRenderer line;
    int i;

    private bool isDrawing;
    private Camera cam;

    private LineDrawer lineDrawer;

    public int ui_posthres;

    // Start is called before the first frame update
    void Start()
    {
        isDrawing = false;
        cam = Camera.main;
        mousePoints = new List<Vector3>();
        stopIndex = new List<int>();
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetMouseButton(0))
        {
            if(isDrawing && Input.mousePosition.y>ui_posthres){
                Vector3 mousePos = Input.mousePosition;
                Vector3 drawPos = cam.ScreenToWorldPoint(new Vector3(mousePos.x, mousePos.y,cam.nearClipPlane + 0.05f));
                if (lineDrawer == null && Input.GetMouseButtonDown(0)){
                    var newDrawer = Instantiate(lineDrawerPrefab);
                    lineDrawer = newDrawer.GetComponent<LineDrawer>();
                    lineDrawer.InitializeLine(transform, drawPos);
                }
                else if(lineDrawer!=null)
                {
                    lineDrawer.AddPoint(drawPos);
                }
                mousePoints.Add(mousePos);
            }   
        }else if(Input.GetMouseButtonDown(1) && Input.mousePosition.y>ui_posthres){ // right mouse click stop drawing
            isDrawing = false;
        }else if(Input.GetMouseButtonUp(0)&& mousePoints.Count>0){
            stopIndex.Add(mousePoints.Count-1);
        }     
    }

    public void activateDraw(){
        isDrawing = true;
        if(lineDrawer!=null) Destroy(lineDrawer.transform.gameObject);
        if(mousePoints.Count>0) mousePoints.Clear();
        if(stopIndex.Count>0) stopIndex.Clear();
    }

    public void stopDraw(){
        isDrawing = false;
    }

    public void ChangeLineView(bool line_visible){
        if(lineDrawer!=null){
            lineDrawer.gameObject.SetActive(line_visible);
        }
    }

}
