//
//  ViewController.swift
//  BlazeFace CoreML
//
//  Created by Vidur Satija on 07/09/20.
//  Copyright © 2020 Vidur Satija. All rights reserved.
//

import AVFoundation
import UIKit
import Vision

class FaceDetectionViewController: UIViewController {
    @IBOutlet var faceView: FaceOverlayView!
  
    let session = AVCaptureSession()
    var previewLayer: AVCaptureVideoPreviewLayer!

    let dataOutputQueue = DispatchQueue(
        label: "video",
        qos: .userInitiated,
        attributes: [],
        autoreleaseFrequency: .workItem)
    
    let bfm = BlazeFaceModel()

    override func viewDidLoad() {
        super.viewDidLoad()
        configureCaptureSession()

        session.startRunning()
    }
}

// MARK: - Video Stuff

extension FaceDetectionViewController {
    func configureCaptureSession() {
    // Define the capture device we want to use
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                   for: .video,
                                                   position: .front) else {
                                                    fatalError("No front video camera available")
        }

        // Connect the camera to the capture session input
        do {
            let cameraInput = try AVCaptureDeviceInput(device: camera)
            session.addInput(cameraInput)
        } catch {
            fatalError(error.localizedDescription)
        }

        // Create the video data output
        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: dataOutputQueue)
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]

        // Add the video output to the capture session
        session.addOutput(videoOutput)

        let videoConnection = videoOutput.connection(with: .video)
        videoConnection?.videoOrientation = .portrait
        videoConnection?.isVideoMirrored = true

        // Configure the preview layer
        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.frame = view.bounds
        view.layer.insertSublayer(previewLayer, at: 0)
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate methods

extension FaceDetectionViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        let faces = bfm.predict(for: imageBuffer)
        DispatchQueue.main.async {
            self.faceView.clear()
            for f in faces {
//                fL.lowHalf *= Double(self.faceView.frame.width)
//                fL.highHalf *= Double(self.faceView.frame.height)
                self.faceView.boundingBox.append(CGRect(x: f.landmark[Face.minBox].x*Double(self.faceView.frame.width), y: f.landmark[Face.minBox].y*Double(self.faceView.frame.height), width: (f.landmark[Face.maxBox].x - f.landmark[Face.minBox].x)*Double(self.faceView.frame.width), height: (f.landmark[Face.maxBox].y - f.landmark[Face.minBox].y)*Double(self.faceView.frame.height)))

                self.faceView.rightEye.append(f.landmark[Face.rightEye])
                self.faceView.leftEye.append(f.landmark[Face.leftEye])
                self.faceView.nose.append(f.landmark[Face.nose])
                self.faceView.mouth.append(f.landmark[Face.mouth])
                self.faceView.rightEar.append(f.landmark[Face.rightEar])
                self.faceView.leftEar.append(f.landmark[Face.leftEar])
            }
            self.faceView.setNeedsDisplay()
        }
        // print(faces.count)
    }
}
