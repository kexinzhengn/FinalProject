using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.Xml;
using System.IO;
using System;
using System.Xml.Serialization;

/*
* Written by ZHENG Kexin
* Save Customized gesture into XML file
*/
[XmlRoot("Gesture")]
public class GestureXML{
    [XmlAttribute]
    public string Name { get; set; }
    [XmlAttribute]
    public int NumPts { get; set; }
    [XmlElement("Point")]
    public List<PointNode> points {get; set;}

    public GestureXML(){
        points = new List<PointNode>();
    }

    public GestureXML(string nme, Vector3[] pts){
        Name = nme;
        NumPts = pts.Length;
        points = new List<PointNode>();
        for(int i = 0; i < pts.Length;i++){
            points.Add(new PointNode(pts[i]));
        }
    }
}

[XmlRoot("Point")]
public class PointNode{
    [XmlAttribute]
    public float X { get; set; }
    [XmlAttribute]
    public float Y { get; set; }
    public PointNode(){

    }
    public PointNode(Vector3 pt){
        X = pt.x;
        Y = pt.y;
    }
}

public class SaveGestureController : MonoBehaviour
{
    // for sketch line on screen
    private DrawLine line_control;

    public Button drawButton;
    public Button saveButton;
    public Button recogButton;
    public TMPro.TextMeshProUGUI resultText;
     public float cammove_scale;
    public float camrot_scale;

    public int saved_num;
    public string saved_name;
    // Start is called before the first frame update
    DollarRecognizer recognizer;
    void Start()
    {
        line_control = transform.GetComponent<DrawLine>();
        drawButton.onClick.AddListener(ActivateDraw);
        saveButton.onClick.AddListener(SaveStrokeAsXML);
        recogButton.onClick.AddListener(StartRecognition);
        ReloadRecognizer();
        //saved_num = 0;
        
    }

    // Update is called once per frame
    void Update()
    {
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

    void SaveStrokeAsXML(){
        string motion_type_name = saved_name;

        var drawnPoints = line_control.MOUSE_PTS;
        var resampled_pts = SystemUtils.ResamplePoints(drawnPoints,1.0f);
        var testges = new GestureXML(motion_type_name+saved_num.ToString(),resampled_pts);
        XmlSerializer serializer = new XmlSerializer(typeof(GestureXML));
        var filePath = Application.streamingAssetsPath+"/CustomGestures";
		if(Directory.Exists(filePath)){
			string files = filePath + "/"+motion_type_name+saved_num.ToString()+".xml";
            FileStream xmlWriter = new FileStream(files, FileMode.Create);
            serializer.Serialize(xmlWriter, testges);
            xmlWriter.Close();
            resultText.text = "Saved" + motion_type_name+saved_num.ToString()+".xml";
            saved_num ++;
        }
        ReloadRecognizer();
    }

    void StartRecognition(){
        var drawnPoints = line_control.MOUSE_PTS;
        var resampled_pts = SystemUtils.ResamplePoints(drawnPoints,1.0f);
        var resampled_2d = SystemUtils.Vector3To2(resampled_pts);
        var res = recognizer.Recognize(resampled_2d);
        resultText.text = res.Match.Name+"  " + res.Score;
    }

    void ActivateDraw(){
        line_control.activateDraw();
    }

    void ReloadRecognizer(){
        recognizer = new DollarRecognizer();
    }
}
