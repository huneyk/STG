# Standard library imports
import json
import os
import sys
import logging
import warnings
from textwrap import dedent
from typing import Optional, Type, List, Dict, Any, Tuple

# Third-party imports
from crewai import Agent, Task, Crew
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import cv2
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox
import urllib.request
import mediapipe as mp
import shutil
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
import absl.logging

app = FastAPI()

# 경고 메시지 숨기기
logging.getLogger('mediapipe').setLevel(logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)

# OpenCV 경고 숨기기
def suppress_qt_warnings():
    os.environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    os.environ["QT_SCALE_FACTOR"] = "1"

suppress_qt_warnings()

# Response 모델 정의
class ImageAnalysisResponse(BaseModel):
    face_position: Dict[str, int]
    eyes_position: Dict[str, Any]
    landmarks: Dict[str, tuple]
    confidence: float
    image_size: Dict[str, int]

# Image Upload Tool
class ImageUploadInput(BaseModel):
    """Image upload tool input."""
    file_path: Optional[str] = Field(description="Path to the image file", default=None)

class ImageUploadTool(BaseTool):
    name: str = "image_upload"
    description: str = "Upload and validate an image file"
    args_schema: Type[BaseModel] = ImageUploadInput

    def _run(self, file_path: Optional[str] = None):
        try:
            # Open file dialog if no path provided
            if not file_path:
                from tkinter import Tk, filedialog
                root = Tk()
                root.withdraw()  # Hide the main window
                file_path = filedialog.askopenfilename(
                    title="Select Image File",
                    filetypes=[
                        ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                        ("All files", "*.*")
                    ]
                )
                if not file_path:
                    return {"error": "No file selected"}

            # Validate image file
            try:
                img = Image.open(file_path)
                img.verify()  # Verify it's a valid image
                return {
                    "status": "success",
                    "file_path": file_path,
                    "format": img.format,
                    "size": img.size
                }
            except Exception as e:
                return {"error": f"Invalid image file: {str(e)}"}

        except Exception as e:
            return {"error": f"Error uploading image: {str(e)}"}

    def _arun(self, file_path: Optional[str] = None):
        # Async implementation if needed
        raise NotImplementedError("Async version not implemented")

# Tool Definitions
class FaceDetectionInput(BaseModel):
    """Face detection tool input."""
    image_path: str = Field(description="Path to the image file to analyze")

class FacialLandmarksInput(BaseModel):
    """Facial landmarks detection tool input."""
    image_path: str = Field(description="Path to the image file to analyze")
    face_location: List[int] = Field(description="Location of detected face in image")



class ReportGeneratorInput(BaseModel):
    """Report generator tool input."""
    analysis_results: Dict[str, Any] = Field(description="All analysis results to include in report")

# Facial Landmarks Tool
class FacialLandmarksInput(BaseModel):
    """Facial landmarks detection tool input."""
    image_path: str = Field(description="Path to the image file to analyze")
    face_location: list = Field(description="Location of detected face in image")

class FacialLandmarksTool(BaseTool):
    name: str = "facial_landmarks_detection"
    description: str = "Detect facial landmarks including eyes, nose, mouth, and jawline"
    args_schema: Type[BaseModel] = FacialLandmarksInput

    def _run(self, image_path: str, face_location: list):
        try:
            # MediaPipe Face Mesh 초기화
            mp_face_mesh = mp.solutions.face_mesh
            with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            ) as face_mesh:
                # 이미지 로드 및 처리
                image = cv2.imread(image_path)
                if image is None:
                    raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image_rgb)
                
                if not results.multi_face_landmarks:
                    raise ValueError("얼굴을 찾을 수 없습니다.")
                
                landmarks = results.multi_face_landmarks[0]
                h, w = image.shape[:2]
                
                # 랜드마크 분석 결과 생성
                landmarks_dict = {
                    "face_shape": self._analyze_face_shape(landmarks, w, h),
                    "features": {
                        "eyes": {
                            "left": {
                                "center": [int(landmarks.landmark[33].x * w), int(landmarks.landmark[33].y * h)],
                                "width": int(abs(landmarks.landmark[133].x - landmarks.landmark[33].x) * w),
                                "height": int(abs(landmarks.landmark[145].y - landmarks.landmark[159].y) * h)
                            },
                            "right": {
                                "center": [int(landmarks.landmark[263].x * w), int(landmarks.landmark[263].y * h)],
                                "width": int(abs(landmarks.landmark[263].x - landmarks.landmark[362].x) * w),
                                "height": int(abs(landmarks.landmark[374].y - landmarks.landmark[386].y) * h)
                            }
                        },
                        "nose": {
                            "bridge": [
                                [int(landmarks.landmark[168].x * w), int(landmarks.landmark[168].y * h)],
                                [int(landmarks.landmark[6].x * w), int(landmarks.landmark[6].y * h)]
                            ],
                            "tip": [int(landmarks.landmark[4].x * w), int(landmarks.landmark[4].y * h)]
                        },
                        "mouth": {
                            "corners": [
                                [int(landmarks.landmark[61].x * w), int(landmarks.landmark[61].y * h)],
                                [int(landmarks.landmark[291].x * w), int(landmarks.landmark[291].y * h)]
                            ],
                            "lips": {
                                "upper": [[int(landmarks.landmark[13].x * w), int(landmarks.landmark[13].y * h)]],
                                "lower": [[int(landmarks.landmark[14].x * w), int(landmarks.landmark[14].y * h)]]
                            }
                        },
                        "jawline": [
                            [int(landmarks.landmark[132].x * w), int(landmarks.landmark[132].y * h)],
                            [int(landmarks.landmark[152].x * w), int(landmarks.landmark[152].y * h)],
                            [int(landmarks.landmark[361].x * w), int(landmarks.landmark[361].y * h)]
                        ]
                    },
                    "measurements": {
                        "face_width": int(abs(landmarks.landmark[352].x - landmarks.landmark[123].x) * w),
                        "face_height": int(abs(landmarks.landmark[10].y - landmarks.landmark[152].y) * h),
                        "eye_distance": int(abs(landmarks.landmark[263].x - landmarks.landmark[33].x) * w)
                    }
                }
                
                print("\nFacialLandmarksTool - landmarks:", landmarks_dict)
                return landmarks_dict
                
        except Exception as e:
            print(f"랜드마크 검출 중 오류 발생: {str(e)}")
            return {"error": str(e)}

    def _analyze_face_shape(self, landmarks, w, h):
        # 실제 구현에서는 이미지 분석을 통한 얼굴형 판단
        return "oval"  # 예시 반환값

# Personal Color Tool
class ColorAnalysisInput(BaseModel):
    """Color analysis tool input."""
    image_path: str = Field(description="Path to the image file")
    face_landmarks: dict = Field(description="Detected facial landmarks")

class PersonalColorTool(BaseTool):
    name: str = "personal_color_analysis"
    description: str = "Analyze and determine personal color season"
    args_schema: Type[BaseModel] = ColorAnalysisInput

    def _run(self, image_path: str, face_landmarks: dict):
        try:
            image = cv2.imread(image_path)
            
            # 피부톤 분석
            skin_tone = self._analyze_skin_tone(image, face_landmarks)
            
            # 퍼스널 컬러 시즌 판단
            season_analysis = self._determine_season(skin_tone)
            
            # 컬러 팔레트 생성
            color_palette = self._generate_color_palette(season_analysis)
            
            return {
                "season": season_analysis["season"],
                "characteristics": season_analysis["characteristics"],
                "skin_tone": skin_tone,
                "color_palette": color_palette,
                "seasonal_keywords": season_analysis["keywords"]
            }
        except Exception as e:
            return {"error": str(e)}

    def _analyze_skin_tone(self, image, landmarks):
        # 실제 구현에서는 피부톤 영역 추출 및 분석
        return {
            "base": "warm",
            "undertone": "yellow",
            "brightness": "medium",
            "contrast": "high"
        }

    def _determine_season(self, skin_tone):
        # 실제 구현에서는 피부톤 기반 시즌 판단
        return {
            "season": "Autumn",
            "characteristics": ["Warm", "Muted", "Deep"],
            "keywords": ["Earth", "Rich", "Warm"]
        }

    def _generate_color_palette(self, season_analysis):
        # 시즌별 컬러 팔레트 정의
        season_palettes = {
            "Spring": {
                "base": ["#FFE4C4", "#F5DEB3"],
                "main": ["#FF6B6B", "#4ECDC4"],
                "accent": ["#FFD93D", "#95E1D3"]
            },
            "Summer": {
                "base": ["#F0F4F7", "#E8EDF2"],
                "main": ["#779ECB", "#C3AED6"],
                "accent": ["#FFB6B9", "#8AC6D1"]
            },
            "Autumn": {
                "base": ["#DAA520", "#CD853F"],
                "main": ["#8B4513", "#A0522D"],
                "accent": ["#D2691E", "#B8860B"]
            },
            "Winter": {
                "base": ["#F0F8FF", "#FFFFFF"],
                "main": ["#000080", "#4B0082"],
                "accent": ["#DC143C", "#800080"]
            }
        }
        return season_palettes.get(season_analysis["season"])


    
def process_image(image_path: str) -> Tuple[np.ndarray, mp.solutions.face_mesh.FaceMesh]:
    """MediaPipe를 사용하여 이미지를 처리합니다."""
    try:
        # MediaPipe Face Mesh 초기화
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")

        # BGR을 RGB로 변환 및 처리
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            raise ValueError("얼굴을 찾을 수 없습니다.")

        return image, results.multi_face_landmarks[0]

    except Exception as e:
        raise ValueError(f"이미지 처리 중 오류 발생: {str(e)}")

def detect_face_landmarks(image: np.ndarray) -> dict:
    """MediaPipe를 사용하여 얼굴 특징점을 검출합니다."""
    try:
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            results = face_mesh.process(image_rgb)
            
            if not results.multi_face_landmarks:
                raise ValueError("얼굴을 찾을 수 없습니다.")
            
            landmarks = results.multi_face_landmarks[0]
            
            landmarks_dict = {
                # 눈
                "point_left_eye": (int(landmarks.landmark[33].x * w), int(landmarks.landmark[33].y * h)),
                "point_right_eye": (int(landmarks.landmark[263].x * w), int(landmarks.landmark[263].y * h)),
                
                # 코 랜드마크 수정
                "point_nose_top": (int(landmarks.landmark[168].x * w), int(landmarks.landmark[168].y * h)),    # 콧대 시작점
                "point_nose_tip": (int(landmarks.landmark[4].x * w), int(landmarks.landmark[4].y * h)),        # 코끝
                "point_nose": (int(landmarks.landmark[4].x * w), int(landmarks.landmark[4].y * h)),           # 기존 코끝 포인트 유지
                
                # 입술 (MediaPipe Face Mesh의 입술 랜드마크 수정)
                "point_mouth_left": (int(landmarks.landmark[61].x * w), int(landmarks.landmark[61].y * h)),      # 입술 좌측
                "point_mouth_right": (int(landmarks.landmark[291].x * w), int(landmarks.landmark[291].y * h)),   # 입술 우측
                "point_upper_lip": (int(landmarks.landmark[13].x * w), int(landmarks.landmark[13].y * h)),       # 윗입술 중앙
                "point_lower_lip": (int(landmarks.landmark[14].x * w), int(landmarks.landmark[14].y * h)),       # 아랫입술 중앙
                
                # 입술 높이 측정을 위한 추가 랜드마크
                "point_upper_lip_top": (int(landmarks.landmark[37].x * w), int(landmarks.landmark[37].y * h)),      # 윗입술 상단 중앙
                "point_lower_lip_bottom": (int(landmarks.landmark[84].x * w), int(landmarks.landmark[84].y * h)),   # 아랫입술 하단 중앙
                "point_upper_lip_outer": (int(landmarks.landmark[0].x * w), int(landmarks.landmark[0].y * h)),      # 윗입술 외곽선 중앙
                "point_lower_lip_outer": (int(landmarks.landmark[17].x * w), int(landmarks.landmark[17].y * h)),    # 아랫입술 외곽선 중앙
                
                # 얼굴 윤곽
                "point_chin": (int(landmarks.landmark[152].x * w), int(landmarks.landmark[152].y * h)),
                "point_forehead": (int(landmarks.landmark[10].x * w), int(landmarks.landmark[10].y * h)),
                "point_temple_left": (int(landmarks.landmark[139].x * w), int(landmarks.landmark[139].y * h)),
                "point_temple_right": (int(landmarks.landmark[368].x * w), int(landmarks.landmark[368].y * h)),
                "point_cheek_left": (int(landmarks.landmark[123].x * w), int(landmarks.landmark[123].y * h)),
                "point_cheek_right": (int(landmarks.landmark[352].x * w), int(landmarks.landmark[352].y * h)),
                "point_jaw_left": (int(landmarks.landmark[132].x * w), int(landmarks.landmark[132].y * h)),
                "point_jaw_right": (int(landmarks.landmark[361].x * w), int(landmarks.landmark[361].y * h))
            }
            
            # 디버깅을 위한 랜드마크 출력
            print("\n코 랜드마크 좌표:")
            for key, value in landmarks_dict.items():
                if 'nose' in key:
                    print(f"{key}: {value}")
            
            return landmarks_dict
            
    except Exception as e:
        raise ValueError(f"랜드마크 검출 중 오류 발생: {str(e)}")

def analyze_beauty_features(landmarks: dict, image: np.ndarray) -> Dict[str, Any]:
    """통합된 얼굴 분석을 수행합니다."""
    # 입술 분석
    lips_analysis = analyze_lips(landmarks)
    landmarks = lips_analysis["landmarks"]
    
    return {
        "face_shape": {
            "main_shape": determine_face_shape(landmarks),
            "details": analyze_face_shape_details(landmarks)
        },
        "facial_features": {
            "eyes": analyze_eyes(landmarks),
            "nose": analyze_nose(landmarks),
            "lips": lips_analysis,
            "jaw": analyze_jaw(landmarks),
            "forehead": analyze_forehead(landmarks)
        },
        "proportions": {
            "facial_thirds": analyze_facial_thirds(landmarks),
            "facial_fifths": analyze_facial_fifths(landmarks),
            "golden_ratio": analyze_golden_ratio(landmarks)
        },
        "symmetry": {
            "overall": calculate_overall_symmetry(landmarks),
            "features": analyze_feature_symmetry(landmarks)
        }
    }

def analyze_face_shape_details(landmarks: dict) -> Dict[str, Any]:
    """얼굴형의 세부 특징을 분석합니다."""
    # 얼굴 너비 계산 (관자놀이, 광대, 턱)
    temple_width = calculate_distance(landmarks["point_temple_left"], landmarks["point_temple_right"])
    cheek_width = calculate_distance(landmarks["point_cheek_left"], landmarks["point_cheek_right"])
    jaw_width = calculate_distance(landmarks["point_jaw_left"], landmarks["point_jaw_right"])
    
    # 얼굴 높이 계산
    face_height = calculate_distance(landmarks["point_forehead"], landmarks["point_chin"])
    
    # 얼굴형 특징 분석
    shape_details = {
        "width_ratio": {
            "temple_to_cheek": temple_width / cheek_width,
            "cheek_to_jaw": cheek_width / jaw_width
        },
        "height_to_width": face_height / cheek_width,
        "characteristics": []
    }
    
    # 특징 판단
    if shape_details["width_ratio"]["temple_to_cheek"] > 1.1:
        shape_details["characteristics"].append("이마가 넓은 형")
    if shape_details["width_ratio"]["cheek_to_jaw"] > 1.2:
        shape_details["characteristics"].append("V라인")
    if shape_details["height_to_width"] > 1.5:
        shape_details["characteristics"].append("긴 얼굴형")
    elif shape_details["height_to_width"] < 1.2:
        shape_details["characteristics"].append("짧은 얼굴형")
    print ("\nshape_details :" , shape_details)
    return shape_details

def analyze_eyes(landmarks: dict) -> Dict[str, Any]:
    """눈의 특징을 분석합니다."""
    eye_distance = calculate_distance(landmarks["point_left_eye"], landmarks["point_right_eye"])
    face_width = calculate_distance(landmarks["point_cheek_left"], landmarks["point_cheek_right"])
    
    # 눈 사이 간격 분석
    eye_ratio = eye_distance / face_width
    
    eye_analysis = {
        "eye_distance_ratio": eye_ratio,
        "characteristics": []
    }
    
    if eye_ratio > 0.45:
        eye_analysis["characteristics"].append("눈이 멀리 떨어진 편")
    elif eye_ratio < 0.35:
        eye_analysis["characteristics"].append("눈이 가까운 편")
    else:
        eye_analysis["characteristics"].append("표준적인 눈 간격")
    print ("\neye_analysis :" , eye_analysis)
    return eye_analysis

def analyze_nose(landmarks: dict) -> Dict[str, Any]:
    """코의 특징을 분석합니다."""
    try:
        # 전체 코 길이 (콧대 시작점부터 코끝까지)
        nose_length = calculate_distance(landmarks["point_nose_top"], landmarks["point_nose_tip"])
        
        # 코끝부터 윗입술까지의 거리
        nose_to_lip = calculate_distance(landmarks["point_nose_tip"], landmarks["point_upper_lip"])
        
        # 얼굴 세로 길이
        face_height = calculate_distance(landmarks["point_forehead"], landmarks["point_chin"])
        
        # 비율 계산
        nose_ratio = nose_length / face_height if face_height != 0 else 0
        
        # 코 위치 분석
        nose_position = "중간"
        if nose_ratio > 0.4:
            nose_position = "긴"
        elif nose_ratio < 0.3:
            nose_position = "짧은"
            
        nose_level = "적절"
        if nose_ratio > 0.45:
            nose_level = "매우 긴"
        elif nose_ratio < 0.25:
            nose_level = "매우 짧은"
            
        return {
            "measurements": {
                "length": format_float(nose_length),
                "nose_to_lip": format_float(nose_to_lip),
                "ratio": format_float(nose_ratio)
            },
            "nose_position": {
                "position": nose_position,
                "ratio": nose_ratio,
                "level": nose_level
            }
        }
        
    except Exception as e:
        print(f"코 분석 중 오류 발생: {str(e)}")
        return {
            "measurements": {
                "length": 0.00,
                "nose_to_lip": 0.00,
                "ratio": 0.00
            },
            "nose_position": {
                "position": "분석 불가",
                "ratio": 0.00,
                "level": "분석 불가"
            }
        }

def analyze_facial_thirds(landmarks: dict) -> Dict[str, float]:
    """얼굴의 삼정비율을 분석합니다."""
    # 상중하 삼정비율 계산
    upper_third = calculate_distance(landmarks["point_forehead"], landmarks["point_eyebrow_left"])
    middle_third = calculate_distance(landmarks["point_eyebrow_left"], landmarks["point_nose"])
    lower_third = calculate_distance(landmarks["point_nose"], landmarks["point_chin"])
    
    total_height = upper_third + middle_third + lower_third
    print ("\nupper_ratio :" , upper_third / total_height)
    print ("\nmiddle_ratio :" , middle_third / total_height)
    print ("\nlower_ratio :" , lower_third / total_height)
    return {
        "upper_ratio": upper_third / total_height,
        "middle_ratio": middle_third / total_height,
        "lower_ratio": lower_third / total_height
    }

def analyze_golden_ratio(landmarks: dict) -> Dict[str, float]:
    """황금비율 분석을 수행합니다."""
    # 이상적인 황금비율 (1:1.618)
    golden_ratio = 1.618
    
    # 실제 비율 계산
    ratios = {
        "face_height_to_width": (
            calculate_distance(landmarks["point_forehead"], landmarks["point_chin"]) /
            calculate_distance(landmarks["point_cheek_left"], landmarks["point_cheek_right"])
        ),
        "eyes_to_mouth": (
            calculate_distance(landmarks["point_left_eye"], landmarks["point_mouth_left"]) /
            calculate_distance(landmarks["point_mouth_left"], landmarks["point_chin"])
        ),
        "nose_to_chin": (
            calculate_distance(landmarks["point_nose"], landmarks["point_chin"]) /
            calculate_distance(landmarks["point_mouth_left"], landmarks["point_chin"])
        )
    }
    
    # 황금비율과의 차이 계산
    deviations = {
        key: abs(ratio - golden_ratio) / golden_ratio
        for key, ratio in ratios.items()
    }
    print ("\nratios :" , ratios)
    print ("\ndeviations :" , deviations)
    print ("\naverage_deviation :" , sum(deviations.values()) / len(deviations))
    return {
        "ratios": ratios,
        "deviations": deviations,
        "average_deviation": sum(deviations.values()) / len(deviations)
    }

def calculate_distance(point1: tuple, point2: tuple) -> float:
    """두 점 사이의 거리를 계산합니다."""
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def analyze_feature_symmetry(landmarks: dict) -> Dict[str, float]:
    """개별 얼굴 특징의 대칭성을 분석합니다."""
    center_x = landmarks["point_nose"][0]
    
    symmetry_scores = {
        "eyes": 1 - abs(
            (center_x - landmarks["point_left_eye"][0]) -
            (landmarks["point_right_eye"][0] - center_x)
        ) / calculate_distance(landmarks["point_left_eye"], landmarks["point_right_eye"]),
        
        "cheeks": 1 - abs(
            (center_x - landmarks["point_cheek_left"][0]) -
            (landmarks["point_cheek_right"][0] - center_x)
        ) / calculate_distance(landmarks["point_cheek_left"], landmarks["point_cheek_right"]),
        
        "jaw": 1 - abs(
            (center_x - landmarks["point_jaw_left"][0]) -
            (landmarks["point_jaw_right"][0] - center_x)
        ) / calculate_distance(landmarks["point_jaw_left"], landmarks["point_jaw_right"])
    }
    print ("\nsymmetry_scores :" , symmetry_scores)
    return symmetry_scores

def calculate_overall_symmetry(landmarks: dict) -> float:
    """얼굴의 대칭성을 계산합니다."""
    try:
        left_eye = np.array(landmarks["point_left_eye"])
        right_eye = np.array(landmarks["point_right_eye"])
        nose = np.array(landmarks["point_nose"])  # point_nose_bottom 대신 point_nose 사용
        
        left_dist = np.linalg.norm(left_eye - nose)
        right_dist = np.linalg.norm(right_eye - nose)
        
        symmetry = 1 - abs(left_dist - right_dist) / max(left_dist, right_dist)
        return float(symmetry)
    except Exception as e:
        print(f"대칭성 계산 중 오류 발생: {str(e)}")
        return 0.0

def calculate_face_ratio(landmarks: dict) -> float:
    """얼굴의 가로 세로 비율을 계산합니다."""
    left = np.array(landmarks["point_left_eye"])
    right = np.array(landmarks["point_right_eye"])
    top = np.array(landmarks["point_forehead"])
    chin = np.array(landmarks["point_chin"])
    
    width = np.linalg.norm(right - left)
    height = np.linalg.norm(chin - top)
    print ("\nwidth, height :" , width, height)
    print ("\n얼굴의 가로 세로 비율 width / height :" , width / height)
    return float(width / height) if height != 0 else 0

def determine_face_shape(landmarks: dict) -> str:
    """얼굴형을 분석합니다."""
    # 얼굴 너비 계산
    face_width = np.linalg.norm(
        np.array(landmarks["point_cheek_left"]) - 
        np.array(landmarks["point_cheek_right"]))
        
    face_height = np.linalg.norm(
        np.array(landmarks["point_forehead"]) - 
        np.array(landmarks["point_chin"]))
        
    jaw_width = np.linalg.norm(
        np.array(landmarks["point_jaw_left"]) - 
        np.array(landmarks["point_jaw_right"]))
        
    forehead_width = np.linalg.norm(
        np.array(landmarks["point_temple_left"]) - 
        np.array(landmarks["point_temple_right"]))
    
    # 비율에 따른 얼굴형 판단
    ratio = face_width / face_height
    jaw_ratio = jaw_width / face_width
    forehead_ratio = forehead_width / face_width
    
    if ratio > 0.9:
        if jaw_ratio < 0.8:
            print("\n하트형")
            return "하트형"
        else:
            return "둥근형"
    else:
        if jaw_ratio < 0.75:
            if forehead_ratio > 0.9:
                print("\n역삼각형")
                return "역삼각형"
            else:
                print("\n계란형")
                return "계란형"
        else:
            if forehead_ratio > 0.9:
                print("\n사각형")
                return "사각형"
            else:
                print("\n긴 얼굴형")
                return "긴 얼굴형"

def analyze_jaw_line(landmarks: dict) -> str:
    """턱선을 분석합니다."""
    jaw_angle = calculate_angle(
        landmarks["point_jaw_left"],
        landmarks["point_chin"],
        landmarks["point_jaw_right"]
    )
    
    if jaw_angle < 80:
        print ("\n뾰족한 턱선") 
        return "뾰족한 턱선", jaw_angle
    elif jaw_angle < 100:
        print ("\n부드러운 V라인")
        return "부드러운 V라인", jaw_angle
    else:
        print ("\n둥근 턱선")
        return "둥근 턱선", jaw_angle

def analyze_forehead(landmarks: dict) -> str:
    """이마 형태를 분석합니다."""
    forehead_width = np.linalg.norm(
        np.array(landmarks["point_temple_left"]) - 
        np.array(landmarks["point_temple_right"])
    )
    face_width = np.linalg.norm(
        np.array(landmarks["point_cheek_left"]) - 
        np.array(landmarks["point_cheek_right"])
    )
    
    ratio = forehead_width / face_width
    if ratio > 1.1:
        print ("\n넓은 이마")
        return "넓은 이마"
    elif ratio < 0.9:
        print ("\n좁은 이마")
        return "좁은 이마"
    else:
        print ("\n균형잡힌 이마")
        return "균형잡힌 이마"

def format_float(value: float) -> float:
    """소수점 2자리로 포맷팅합니다."""
    return float(format(value, '.2f'))

def analyze_lips(landmarks: dict) -> Dict[str, Any]:
    """입술의 특징을 분석합니다."""
    try:
        # 기본 측정값 계산
        face_width = calculate_distance(
            landmarks["point_cheek_left"], 
            landmarks["point_cheek_right"]
        )
        face_height = calculate_distance(
            landmarks["point_forehead"], 
            landmarks["point_chin"]
        )
        
        # 입술 측정 시도
        try:
            lip_width = calculate_distance(
                landmarks["point_mouth_left"], 
                landmarks["point_mouth_right"]
            )
        except:
            print("입술 너비 측정 실패")
            lip_width = 0
            
        try:
            lip_height = calculate_distance(
                landmarks["point_upper_lip_top"], 
                landmarks["point_lower_lip_bottom"]
            )

            print ("\n\npoint_upper_lip_top :" , landmarks["point_upper_lip_top"])
            print ("\npoint_lower_lip_bottom :" , landmarks["point_lower_lip_bottom"])
            
        except:
            print("입술 높이 측정 실패")
            lip_height = 0
        
        # 비율 계산 (0으로 나누기 방지)
        width_ratio = lip_width / face_width if face_width != 0 else 0
        height_ratio = lip_height / face_height if face_height != 0 else 0
        
        # 대칭성 분석 시도
        try:
            left_half = calculate_distance(
                landmarks["point_mouth_left"], 
                landmarks["point_upper_lip"]
            )
            right_half = calculate_distance(
                landmarks["point_mouth_right"], 
                landmarks["point_upper_lip"]
            )
            symmetry = 1 - abs(left_half - right_half) / max(left_half, right_half) if max(left_half, right_half) != 0 else 0
        except:
            print("입술 대칭성 측정 실패")
            symmetry = 0
        
        # 특징 분석
        characteristics = []
        
        # 유효한 측정값이 있을 경우만 특징 분석
        if lip_width > 0 and face_width > 0:
            if width_ratio > 0.45:
                characteristics.append("넓은 입술")
            elif width_ratio < 0.35:
                characteristics.append("좁은 입술")
            else:
                characteristics.append("균형잡힌 입술 너비")
        
        if lip_height > 0 and face_height > 0:
            if height_ratio > 0.08:
                characteristics.append("두꺼운 입술")
            elif height_ratio < 0.05:
                characteristics.append("얇은 입술")
            else:
                characteristics.append("적당한 두께의 입술")
        
        if symmetry > 0:
            if symmetry > 0.95:
                characteristics.append("대칭적인 입술")
            elif symmetry < 0.85:
                characteristics.append("비대칭적인 입술")
        
        # 특징이 하나도 없으면 분석 불가 표시
        if not characteristics:
            characteristics = ["분석 불가"]
        
        return {
            "measurements": {
                "width": format_float(lip_width),
                "height": format_float(lip_height),
                "width_ratio": format_float(width_ratio),
                "height_ratio": format_float(height_ratio),
                "symmetry": format_float(symmetry)
            },
            "characteristics": characteristics,
            "landmarks": landmarks
        }
        
    except Exception as e:
        print(f"입술 분석 중 오류 발생: {str(e)}")
        return {
            "measurements": {
                "width": 0.00,
                "height": 0.00,
                "width_ratio": 0.00,
                "height_ratio": 0.00,
                "symmetry": 0.00
            },
            "characteristics": ["분석 불가"],
            "landmarks": landmarks
        }

def analyze_jaw(landmarks: dict) -> Dict[str, Any]:
    jaw_width = calculate_distance(landmarks["point_jaw_left"], landmarks["point_jaw_right"])
    face_width = calculate_distance(landmarks["point_cheek_left"], landmarks["point_cheek_right"])
    return {"jaw_ratio": jaw_width / face_width}

def analyze_forehead(landmarks: dict) -> Dict[str, Any]:
    forehead_width = calculate_distance(landmarks["point_temple_left"], landmarks["point_temple_right"])
    face_width = calculate_distance(landmarks["point_cheek_left"], landmarks["point_cheek_right"])
    return {"forehead_ratio": forehead_width / face_width}

def analyze_facial_fifths(landmarks: dict) -> Dict[str, Any]:
    """얼굴의 오정비율을 분석합니다."""
    try:
        face_width = calculate_distance(landmarks["point_temple_left"], landmarks["point_temple_right"])
        fifth_width = face_width / 5
        
        outer_eye_left = calculate_distance(landmarks["point_temple_left"], landmarks["point_left_eye"])
        outer_eye_right = calculate_distance(landmarks["point_right_eye"], landmarks["point_temple_right"])
        eye_width_left = calculate_distance(landmarks["point_left_eye"], landmarks["point_nose"])
        eye_width_right = calculate_distance(landmarks["point_nose"], landmarks["point_right_eye"])
        intercanthal = calculate_distance(landmarks["point_left_eye"], landmarks["point_right_eye"])
        
        ratios = {
            "outer_eye_left_ratio": format_float(outer_eye_left / fifth_width),
            "eye_width_left_ratio": format_float(eye_width_left / fifth_width),
            "intercanthal_ratio": format_float(intercanthal / fifth_width),
            "eye_width_right_ratio": format_float(eye_width_right / fifth_width),
            "outer_eye_right_ratio": format_float(outer_eye_right / fifth_width)
        }
        
        deviations = {
            key: format_float(abs(ratio - 1.0)) for key, ratio in ratios.items()
        }
        
        evaluation = []
        if max(deviations.values()) < 0.15:
            evaluation.append("이상적인 오정비율")
        else:
            if deviations["intercanthal_ratio"] > 0.15:
                if ratios["intercanthal_ratio"] > 1:
                    evaluation.append("눈 사이 간격이 넓은 편")
                else:
                    evaluation.append("눈 사이 간격이 좁은 편")
            
            if deviations["outer_eye_left_ratio"] > 0.15 or deviations["outer_eye_right_ratio"] > 0.15:
                evaluation.append("측면부 비율이 불균형")
        
        return {
            "ratios": ratios,
            "deviations": deviations,
            "evaluation": evaluation,
            "ideal_fifth_width": format_float(fifth_width)
        }
        
    except Exception as e:
        print(f"오정비율 분석 중 오류 발생: {str(e)}")
        return {
            "ratios": {},
            "deviations": {},
            "evaluation": ["분석 불가"],
            "ideal_fifth_width": 0.0
        }

def calculate_angle(p1, p2, p3) -> float:
    """세 점 사이의 각도를 계산합니다."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    print ("\nangle :" , np.degrees(angle))
    return np.degrees(angle)

def browse_image() -> str:
    """파일 브라우저를 열어 이미지 파일을 선택합니다."""
    root = tk.Tk()
    root.withdraw()  # tkinter 창 숨기기
    
    file_path = filedialog.askopenfilename(
        title="이미지 파일 선택",
        filetypes=[
            ("이미지 파일", "*.jpg *.jpeg *.png *.bmp"),
            ("모든 파일", "*.*")
        ]
    )
    
    if not file_path:
        raise ValueError("파일이 선택되지 않았습니다.")
    
    return file_path

def save_analysis_results(results: Dict[str, Any], output_path: str):
    """분석 결과를 JSON 파일로 저장합니다."""
    try:
        # 저장할 데이터 구성
        output_data = {
            "eyes_position": results["eyes"]["eyes_position"],
            "nose_position": results["nose"]["nose_position"],
            "lips_position": results["lips"]["lips_position"],
            "jaw_angle": results["jaw"]["jaw_angle"],
            "shape_details": results["shape_details"],
            "ratios": results["ratios"],
            "symmetry_scores": results["symmetry_scores"]
        }
        
        # JSON 파일로 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"결과 저장 중 오류 발생: {str(e)}")
        print(f"Results content: {results}")  # 디버깅을 위한 결과 내용 출력

class FaceAnalyzer:
    """얼굴 분석을 위한 메인 클래스"""
    def __init__(self):
        self.face_detector = FaceDetection()
        
    def analyze_face(self, image_path: str) -> Dict[str, Any]:
        """전체 얼굴 분석 프로세스를 실행합니다."""
        try:
            # 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
            
            # 얼굴 검출
            detection_result = self.face_detector.detect_faces(image)
            
            # 특징 분석
            analysis_result = analyze_beauty_features(detection_result["landmarks"], image)
            
            # 결과 시각화
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"analyzed_{os.path.basename(image_path)}")
            self.face_detector.visualize_detection(image, detection_result, output_path)
            
            # 결과 저장
            results = {
                "face_detection": detection_result,
                "analysis": analysis_result,
                "output_image": output_path
            }
            save_analysis_results(results, os.path.join(output_dir, "analysis_results.json"))
            
            return results
            
        except Exception as e:
            raise ValueError(f"얼굴 분석 중 오류 발생: {str(e)}")

class FaceDetection:
    """MediaPipe를 사용한 얼굴 검출 클래스"""
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def detect_faces(self, image: np.ndarray) -> Dict[str, Any]:
        try:
            with self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            ) as face_mesh:
                # BGR을 RGB로 변환
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w = image.shape[:2]
                
                # MediaPipe로 얼굴 랜드마크 검출
                results = face_mesh.process(image_rgb)

                if not results.multi_face_landmarks:
                    raise ValueError("이미지에서 얼굴을 찾을 수 없습니다.")

                landmarks = results.multi_face_landmarks[0]
                
                # 얼굴 영역 계산
                x_coordinates = [landmark.x * w for landmark in landmarks.landmark]
                y_coordinates = [landmark.y * h for landmark in landmarks.landmark]
                
                x_min, x_max = int(min(x_coordinates)), int(max(x_coordinates))
                y_min, y_max = int(min(y_coordinates)), int(max(y_coordinates))
                
                # 눈 관련 랜드마크 추출
                # MediaPipe 눈 랜드마크 인덱스
                left_eye_indices = [33, 133]   # 왼쪽 눈 중심점
                right_eye_indices = [362, 263]  # 오른쪽 눈 중심점
                
                # 눈 위치 및 크기 계산
                left_eye_pos = np.mean([[landmarks.landmark[idx].x * w,
                                       landmarks.landmark[idx].y * h] 
                                      for idx in left_eye_indices], axis=0)
                right_eye_pos = np.mean([[landmarks.landmark[idx].x * w,
                                        landmarks.landmark[idx].y * h] 
                                       for idx in right_eye_indices], axis=0)

                # 눈 크기 계산
                left_eye_width = abs(landmarks.landmark[33].x - landmarks.landmark[133].x) * w
                left_eye_height = abs(landmarks.landmark[159].y - landmarks.landmark[145].y) * h
                right_eye_width = abs(landmarks.landmark[362].x - landmarks.landmark[263].x) * w
                right_eye_height = abs(landmarks.landmark[386].y - landmarks.landmark[374].y) * h

                eyes_position = {
                    "point_left_eye": (int(left_eye_pos[0]), int(left_eye_pos[1])),
                    "point_right_eye": (int(right_eye_pos[0]), int(right_eye_pos[1])),
                    "left_eye_width": int(left_eye_width),
                    "left_eye_height": int(left_eye_height),
                    "right_eye_width": int(right_eye_width),
                    "right_eye_height": int(right_eye_height)
                }

                # 전체 랜드마크 정보 생성
                landmarks_dict = {
                    # 이마
                    "point_forehead": (
                        int(landmarks.landmark[10].x * w),
                        int(landmarks.landmark[10].y * h)
                    ),
                    # 눈
                    "point_left_eye": eyes_position["point_left_eye"],
                    "point_right_eye": eyes_position["point_right_eye"],
                    # 코 랜드마크 수정
                    "point_nose_top": (int(landmarks.landmark[168].x * w), int(landmarks.landmark[168].y * h)),    # 콧대 시작점
                    "point_nose_tip": (int(landmarks.landmark[4].x * w), int(landmarks.landmark[4].y * h)),        # 코끝
                    "point_nose": (int(landmarks.landmark[4].x * w), int(landmarks.landmark[4].y * h)),           # 기존 코끝 포인트 유지
                    # 입
                    "point_mouth_left": (
                        int(landmarks.landmark[61].x * w),
                        int(landmarks.landmark[61].y * h)
                    ),
                    "point_mouth_right": (
                        int(landmarks.landmark[291].x * w),
                        int(landmarks.landmark[291].y * h)
                    ),
                    # 턱
                    "point_chin": (
                        int(landmarks.landmark[152].x * w),
                        int(landmarks.landmark[152].y * h)
                    ),
                    # 볼
                    "point_cheek_left": (
                        int(landmarks.landmark[123].x * w),
                        int(landmarks.landmark[123].y * h)
                    ),
                    "point_cheek_right": (
                        int(landmarks.landmark[352].x * w),
                        int(landmarks.landmark[352].y * h)
                    ),
                    # 턱선
                    "point_jaw_left": (
                        int(landmarks.landmark[132].x * w),
                        int(landmarks.landmark[132].y * h)
                    ),
                    "point_jaw_right": (
                        int(landmarks.landmark[361].x * w),
                        int(landmarks.landmark[361].y * h)
                    ),
                    # 관자놀이
                    "point_temple_left": (
                        int(landmarks.landmark[139].x * w),
                        int(landmarks.landmark[139].y * h)
                    ),
                    "point_temple_right": (
                        int(landmarks.landmark[368].x * w),
                        int(landmarks.landmark[368].y * h)
                    ),
                    # 눈썹
                    "point_eyebrow_left": (
                        int(landmarks.landmark[70].x * w),
                        int(landmarks.landmark[70].y * h)
                    ),
                    "point_eyebrow_right": (
                        int(landmarks.landmark[300].x * w),
                        int(landmarks.landmark[300].y * h)
                    ),
                    # 입술 랜드마크 (MediaPipe Face Mesh의 정확한 입술 포인트)
                    "point_mouth_left": (int(landmarks.landmark[61].x * w), int(landmarks.landmark[61].y * h)),      # 입술 좌측
                    "point_mouth_right": (int(landmarks.landmark[291].x * w), int(landmarks.landmark[291].y * h)),   # 입술 우측
                    "point_upper_lip": (int(landmarks.landmark[13].x * w), int(landmarks.landmark[13].y * h)),       # 윗입술 중앙
                    "point_lower_lip": (int(landmarks.landmark[14].x * w), int(landmarks.landmark[14].y * h)),       # 아랫입술 중앙
                    # 입술 높이 측정을 위한 추가 랜드마크
                    "point_upper_lip_top": (int(landmarks.landmark[37].x * w), int(landmarks.landmark[37].y * h)),      # 윗입술 상단 중앙
                    "point_lower_lip_bottom": (int(landmarks.landmark[84].x * w), int(landmarks.landmark[84].y * h)),   # 아랫입술 하단 중앙
                    "point_upper_lip_outer": (int(landmarks.landmark[0].x * w), int(landmarks.landmark[0].y * h)),      # 윗입술 외곽선 중앙
                    "point_lower_lip_outer": (int(landmarks.landmark[17].x * w), int(landmarks.landmark[17].y * h)),    # 아랫입술 외곽선 중앙
                }

                # 신뢰도 계산: 주요 랜드마크들의 정확도 평가
                key_landmarks = [
                    landmarks.landmark[1],    # 코
                    landmarks.landmark[33],   # 왼쪽 눈
                    landmarks.landmark[263],  # 오른쪽 눈
                    landmarks.landmark[61],   # 입 좌측
                    landmarks.landmark[291],  # 입 우측
                    landmarks.landmark[152],  # 턱
                    landmarks.landmark[10]    # 이마
                ]
                
                # z 좌표의 표준편차를 이용한 신뢰도 계산
                z_coords = [lm.z for lm in key_landmarks]
                z_std = np.std(z_coords)
                confidence = 1.0 / (1.0 + z_std)  # z 좌표가 일정할수록 높은 신뢰도
                
                return {
                    "face_position": {
                        "x": x_min,
                        "y": y_min,
                        "width": x_max - x_min,
                        "height": y_max - y_min
                    },
                    "eyes_position": eyes_position,
                    "landmarks": landmarks_dict,
                    "confidence": float(confidence),  # 새로운 신뢰도 계산 방식
                    "image_size": {
                        "width": w,
                        "height": h
                    }
                }

        except Exception as e:
            raise ValueError(f"얼굴 검출 중 오류 발생: {str(e)}")

    def visualize_detection(self, image: np.ndarray, detection_result: Dict[str, Any], output_path: str) -> None:
        """검출 결과를 시각화하여 저장합니다."""
        try:
            # 이미지 복사
            vis_image = image.copy()
            
            # 얼굴 영역 표시
            pos = detection_result["face_position"]
            cv2.rectangle(vis_image, 
                        (pos["x"], pos["y"]), 
                        (pos["x"]+pos["width"], pos["y"]+pos["height"]), 
                        (0, 255, 0), 2)
            
            # 랜드마크 표시
            for point_name, (px, py) in detection_result["landmarks"].items():
                cv2.circle(vis_image, (px, py), 2, (0, 0, 255), -1)
                cv2.putText(vis_image, point_name, (px, py-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            
            # 결과 저장
            cv2.imwrite(output_path, vis_image)
            
        except Exception as e:
            raise ValueError(f"시각화 중 오류 발생: {str(e)}")

def visualize_landmarks(image: np.ndarray, landmarks: dict, analysis_results: dict = None) -> np.ndarray:
    """랜드마크와 분석 결과를 이미지에 시각화합니다."""
    vis_image = image.copy()
    
    # 색상 정의 (BGR 형식)
    COLORS = {
        'eyes': (255, 128, 0),     # 파란색 계열
        'nose': (0, 255, 0),       # 녹색
        'lips': (0, 0, 255),       # 빨간색
        'face': (255, 255, 0),     # 청록색
        'text': (255, 255, 255),   # 흰색
        'measure': (0, 255, 255),  # 노란색
        'result': (0, 0, 255)      # 빨간색으로 통일
    }
    
    try:
        # 기본 랜드마크 표시
        for key, point in landmarks.items():
            cv2.circle(vis_image, point, 2, COLORS[key], -1)
        
        # 분석 결과 표시 함수
        def draw_position_info(position_key, position_data, center_point, y_offset=0, title=""):
            mid_x = center_point[0]
            mid_y = center_point[1] + y_offset
            
            position_text = f"{title}: {position_data['position']} ({position_data['ratio']:.2f})"
            level_text = f"Level: {position_data['level']}"
            
            # 텍스트 크기 계산
            (text_width, text_height), _ = cv2.getTextSize(position_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # 배경 사각형
            padding = 10
            rect_x1 = mid_x - text_width//2 - padding
            rect_x2 = mid_x + text_width//2 + padding
            rect_y1 = mid_y - 40
            rect_y2 = mid_y + 10
            
            cv2.rectangle(vis_image, 
                        (rect_x1, rect_y1),
                        (rect_x2, rect_y2),
                        (0, 0, 0), -1)
            
            # 텍스트 표시
            cv2.putText(vis_image, position_text,
                      (mid_x - text_width//2, mid_y - 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['result'], 2)
            cv2.putText(vis_image, level_text,
                      (mid_x - text_width//2, mid_y),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['result'], 2)
        
        # 눈 위치 표시
        if analysis_results and 'eyes_position' in analysis_results:
            left_eye = landmarks['point_left_eye']
            right_eye = landmarks['point_right_eye']
            eye_center = ((left_eye[0] + right_eye[0])//2, (left_eye[1] + right_eye[1])//2)
            draw_position_info('eyes_position', 
                             analysis_results['eyes_position'], 
                             eye_center, 
                             y_offset=-50,
                             title="Eyes")
            
            # 눈 수평선
            cv2.line(vis_image, (0, eye_center[1]), (vis_image.shape[1], eye_center[1]), 
                    COLORS['result'], 1)
        
        # 코 위치 표시
        if analysis_results and 'nose_position' in analysis_results:
            nose_top = landmarks['point_nose_top']
            nose_tip = landmarks['point_nose_tip']
            nose_center = ((nose_top[0] + nose_tip[0])//2, (nose_top[1] + nose_tip[1])//2)
            draw_position_info('nose_position', 
                             analysis_results['nose_position'], 
                             nose_center,
                             y_offset=0,
                             title="Nose")
            
            # 코 수직선
            cv2.line(vis_image, nose_top, nose_tip, COLORS['result'], 2)
        
        # 입술 위치 표시
        if analysis_results and 'lips_position' in analysis_results:
            left_mouth = landmarks['point_mouth_left']
            right_mouth = landmarks['point_mouth_right']
            mouth_center = ((left_mouth[0] + right_mouth[0])//2, (left_mouth[1] + right_mouth[1])//2)
            draw_position_info('lips_position', 
                             analysis_results['lips_position'], 
                             mouth_center,
                             y_offset=50,
                             title="Lips")
            
            # 입술 수평선
            cv2.line(vis_image, (0, mouth_center[1]), (vis_image.shape[1], mouth_center[1]), 
                    COLORS['result'], 1)
        
        return vis_image
        
    except Exception as e:
        print(f"랜드마크 시각화 중 오류 발생: {str(e)}")
        return image

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """API 엔드포인트: 이미지 분석"""
    try:
        # 임시 파일로 저장
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 분석 실행
        analyzer = FaceAnalyzer()
        result = analyzer.analyze_face(temp_path)
        
        # 임시 파일 삭제
        os.remove(temp_path)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def main():
    """프로그램 메인 실행 함수"""
    try:
        print("얼굴 분석 프로그램을 시작합니다...")
        
        # 이미지 선택
        image_path = browse_image()
        print(f"선택된 이미지: {image_path}")
        
        # 분석 실행
        analyzer = FaceAnalyzer()
        result = analyzer.analyze_face(image_path)
        
        # 결과 표시
        tk.messagebox.showinfo("분석 완료", 
            f"얼굴 분석이 완료되었습니다.\n"
            f"검출된 얼굴 신뢰도: {result['face_detection']['confidence']:.2f}\n"
            f"결과 이미지: {result['output_image']}"
        )
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        tk.messagebox.showerror("오류", f"분석 중 오류가 발생했습니다:\n{str(e)}")

if __name__ == "__main__":
    main()

