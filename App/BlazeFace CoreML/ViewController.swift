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

    var maxX: CGFloat = 0.0
    var midY: CGFloat = 0.0
    var maxY: CGFloat = 0.0

    override func viewDidLoad() {
        super.viewDidLoad()
        configureCaptureSession()

        maxX = view.bounds.maxX
        midY = view.bounds.midY
        maxY = view.bounds.maxY

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
                var fL = faces.landmarks[0]
                let fC = faces.confidence[0]
                fL.lowHalf *= Double(self.faceView.frame.width)
                fL.highHalf *= Double(self.faceView.frame.height)
                self.faceView.boundingBox = CGRect(x: fL[0], y: fL[8], width: fL[1]-fL[0], height: fL[9]-fL[8])
                self.faceView.setNeedsDisplay()
            }
        } else {
            DispatchQueue.main.async {
                self.faceView.clear()
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
