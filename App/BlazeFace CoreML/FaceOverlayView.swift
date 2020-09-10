//
//  FaceOverlayView.swift
//  BlazeFace CoreML
//
//  Created by Vidur Satija on 07/09/20.
//  Copyright © 2020 Vidur Satija. All rights reserved.
//

import Foundation

import UIKit
import Vision

class FaceOverlayView: UIView {
    var leftEye: [CGPoint] = []
    var rightEye: [CGPoint] = []
    var nose: [CGPoint] = []
    var mouth: [CGPoint] = []
    var leftEar: [CGPoint] = []
    var rightEar: [CGPoint] = []

    var boundingBox: [CGRect] = []

    func clear() {
        leftEye = []
        rightEye = []
        nose = []
        mouth = []
        leftEar = []
        rightEar = []

        boundingBox = []
    }
    override func draw(_ rect: CGRect) {
        guard let context = UIGraphicsGetCurrentContext() else {
          return
        }
        context.saveGState()
        defer {
          context.restoreGState()
        }

        if boundingBox.count > 0 {
            for bb in boundingBox {
                context.addRect(bb)
                UIColor.red.setStroke()
                context.strokePath()
            }
        }
        
        UIColor.white.setStroke()
        for bodyPart in [leftEye, rightEye, nose, mouth, leftEar, rightEar] {
            if bodyPart.count > 0 {
                for bP in bodyPart {
                    context.addEllipse(in: CGRect(x: bP.x, y: bP.y, width: 10.0, height: 10.0))
                    context.closePath()
                    context.strokePath()
                }
            }
        }
    }
}
