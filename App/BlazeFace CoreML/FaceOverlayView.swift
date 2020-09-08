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
    var leftEye: CGPoint = .zero
    var rightEye: CGPoint = .zero
    var nose: CGPoint = .zero

    var boundingBox = CGRect.zero

    func clear() {
        leftEye = .zero
        rightEye = .zero
        nose = .zero

        boundingBox = .zero

        DispatchQueue.main.async {
            self.setNeedsDisplay()
        }
    }
    override func draw(_ rect: CGRect) {
        guard let context = UIGraphicsGetCurrentContext() else {
          return
        }
        context.saveGState()
        defer {
          context.restoreGState()
        }

        context.addRect(boundingBox)
        UIColor.red.setStroke()
        context.strokePath()

        UIColor.white.setStroke()
        if leftEye != .zero {
            context.addEllipse(in: CGRect(x: leftEye.x, y: leftEye.y, width: 2.0, height: 2.0))
            context.closePath()
            context.strokePath()
        }
        
        if rightEye != .zero {
            context.addEllipse(in: CGRect(x: rightEye.x, y: rightEye.y, width: 2.0, height: 2.0))
            context.closePath()
            context.strokePath()
        }
    }
}
