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
        if faces.confidence.count > 0 {
            DispatchQueue.main.async {
                self.faceView.clear()
                for f in faces.landmarks {
                    var fL = f
                    fL.lowHalf *= Double(self.faceView.frame.width)
                    fL.highHalf *= Double(self.faceView.frame.height)
                    self.faceView.boundingBox.append(CGRect(x: fL[0], y: fL[8], width: fL[1]-fL[0], height: fL[9]-fL[8]))
                    self.faceView.rightEye.append(CGPoint(x: fL[2], y: fL[10]))
                    self.faceView.leftEye.append(CGPoint(x: fL[3], y: fL[11]))
                    self.faceView.nose.append(CGPoint(x: fL[4], y: fL[12]))
                    self.faceView.mouth.append(CGPoint(x: fL[5], y: fL[13]))
                    self.faceView.rightEar.append(CGPoint(x: fL[6], y: fL[14]))
                    self.faceView.leftEar.append(CGPoint(x: fL[7], y: fL[15]))
                }
                self.faceView.setNeedsDisplay()
            }
        } else {
            DispatchQueue.main.async {
                self.faceView.clear()
                self.faceView.setNeedsDisplay()
            }
        }
        // print(faces.confidence.count)
    }
}

//extension FaceDetectionViewController {
//    func updateFaceView(for result: VNFaceObservation) {
//        defer {
//            DispatchQueue.main.async {
//                self.faceView.setNeedsDisplay()
//            }
//        }
//    }
//}
