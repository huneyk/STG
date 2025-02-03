# Standard library imports
import json
from textwrap import dedent
from typing import Optional, Type, List, Dict, Any

# Third-party imports
from crewai import Agent, Task, Crew
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import cv2
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import urllib.request
import mediapipe as mp
import shutil
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException

app = FastAPI()


# Image Upload Tool
class ImageUploadInput(BaseModel):
    """Image upload tool input."""
    file_path: Optional[str] = Field(description="Path to the image file", default=None)
'''
class ImageUpload:
    """Image upload tool implementation."""
    
    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = upload_dir
        os.makedirs(upload_dir, exist_ok=True)
    
    async def save_upload_file(self, upload_file: UploadFile) -> str:
        """업로드된 파일을 저장하고 경로를 반환합니다."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = os.path.splitext(upload_file.filename)[1]
        
        if file_extension.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
            raise ValueError("지원하지 않는 파일 형식입니다. (지원 형식: jpg, jpeg, png, bmp)")
        
        file_name = f"image_{timestamp}{file_extension}"
        file_path = os.path.join(self.upload_dir, file_name)
        
        with open(file_path, "wb") as f:
            content = await upload_file.read()
            f.write(content)
        
        return file_path
    
    def cleanup(self, file_path: str) -> None:
        """업로드된 파일을 삭제합니다."""
        if os.path.exists(file_path):
            os.remove(file_path)
'''          

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

class ColorAnalysisInput(BaseModel):
    """Color analysis tool input."""
    image_path: str = Field(description="Path to the image file")
    face_landmarks: Dict[str, Any] = Field(description="Detected facial landmarks")

class MakeupRecommendationInput(BaseModel):
    """Makeup recommendation tool input."""
    personal_color: Dict[str, Any] = Field(description="Personal color analysis results")
    face_features: Dict[str, Any] = Field(description="Facial features information")

class HairstyleRecommendationInput(BaseModel):
    """Hairstyle recommendation tool input."""
    face_shape: str = Field(description="Detected face shape")
    personal_color: Dict[str, Any] = Field(description="Personal color analysis results")

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
            # 실제 구현에서는 dlib 또는 MediaPipe 사용
            image = cv2.imread(image_path)
            landmarks = {
                "face_shape": self._analyze_face_shape(image),
                "features": {
                    "eyes": {
                        "left": {"center": [100, 100], "width": 30, "height": 15},
                        "right": {"center": [170, 100], "width": 30, "height": 15}
                    },
                    "nose": {
                        "bridge": [[135, 90], [135, 120]],
                        "tip": [135, 130]
                    },
                    "mouth": {
                        "corners": [[120, 150], [150, 150]],
                        "lips": {"upper": [[135, 145]], "lower": [[135, 155]]}
                    },
                    "jawline": [[90, 160], [135, 180], [180, 160]]
                },
                "measurements": {
                    "face_width": 180,
                    "face_height": 220,
                    "eye_distance": 70
                }
            }
            print ("\nFacialLandmarksTool - landmarks :" , landmarks)
            return landmarks
        except Exception as e:
            return {"error": str(e)}

    def _analyze_face_shape(self, image):
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

# Makeup Recommendation Tool
class MakeupRecommendationInput(BaseModel):
    """Makeup recommendation tool input."""
    personal_color: dict = Field(description="Personal color analysis results")
    face_features: dict = Field(description="Facial features information")

class MakeupRecommendationTool(BaseTool):
    name: str = "makeup_style_recommendation"
    description: str = "Recommend makeup styles based on personal color and facial features"
    args_schema: Type[BaseModel] = MakeupRecommendationInput

    def _run(self, personal_color: dict, face_features: dict):
        try:
            # 메이크업 스타일 추천 생성
            base_makeup = self._recommend_base_makeup(personal_color)
            eye_makeup = self._recommend_eye_makeup(personal_color, face_features)
            lip_makeup = self._recommend_lip_makeup(personal_color)
            
            return {
                "base_makeup": base_makeup,
                "eye_makeup": eye_makeup,
                "lip_makeup": lip_makeup,
                "application_tips": self._generate_application_tips(face_features),
                "product_recommendations": self._recommend_products(personal_color)
            }
        except Exception as e:
            return {"error": str(e)}

    def _recommend_base_makeup(self, personal_color):
        # 베이스 메이크업 추천 로직
        return {
            "foundation": {
                "tone": "21N",
                "type": "Matte",
                "coverage": "Medium"
            },
            "concealer": {
                "tone": "Light",
                "type": "Creamy",
                "coverage": "High"
            },
            "powder": {
                "type": "Translucent",
                "finish": "Natural"
            }
        }

    def _recommend_eye_makeup(self, personal_color, face_features):
        return {
            "eyeshadow": {
                "main_colors": ["Champagne", "Brown", "Black"],
                "technique": "Gradient",
                "finish": "Shimmer"
            },
            "eyeliner": {
                "color": "Black",
                "style": "Wing"
            },
            "mascara": {
                "color": "Black",
                "type": "Volumizing"
            }
        }

    def _recommend_lip_makeup(self, personal_color):
        return {
            "lipstick": {
                "color": "MLBB",
                "finish": "Semi-matte",
                "application": "Gradation"
            }
        }

# Hairstyle Recommendation Tool
class HairstyleRecommendationInput(BaseModel):
    """Hairstyle recommendation tool input."""
    face_shape: str = Field(description="Detected face shape")
    personal_color: dict = Field(description="Personal color analysis results")

class HairstyleRecommendationTool(BaseTool):
    name: str = "hairstyle_recommendation"
    description: str = "Recommend hairstyles based on face shape and personal color"
    args_schema: Type[BaseModel] = HairstyleRecommendationInput

    def _run(self, face_shape: str, personal_color: dict):
        try:
            # 헤어스타일 추천 생성
            recommended_styles = self._get_hairstyle_recommendations(face_shape)
            color_recommendations = self._get_color_recommendations(personal_color)
            
            return {
                "recommended_styles": recommended_styles,
                "color_recommendations": color_recommendations,
                "styling_tips": self._generate_styling_tips(face_shape),
                "maintenance_guide": self._generate_maintenance_guide()
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_hairstyle_recommendations(self, face_shape):
        # 얼굴형별 추천 스타일
        face_shape_styles = {
            "oval": [
                {
                    "name": "Long Layered Cut",
                    "description": "Soft layers framing the face",
                    "length": "Long",
                    "styling_tips": ["Use volumizing products", "Blow dry with round brush"]
                },
                {
                    "name": "Medium Length Bob",
                    "description": "Classic bob with subtle layers",
                    "length": "Medium",
                    "styling_tips": ["Use smoothing serum", "Straighten with flat iron"]
                }
            ],
            # 다른 얼굴형에 대한 스타일도 정의
        }
        return face_shape_styles.get(face_shape.lower(), [])

    def _get_color_recommendations(self, personal_color):
        return {
            "base_color": "Dark Brown",
            "highlights": "Caramel",
            "lowlights": "Chocolate Brown",
            "maintenance_level": "Medium",
            "frequency": "Every 6-8 weeks"
        }

# Report Generator Tool
class ReportGeneratorInput(BaseModel):
    """Report generator tool input."""
    analysis_results: dict = Field(description="All analysis results to include in report")

class ReportGeneratorTool(BaseTool):
    name: str = "report_template_generation"
    description: str = "Generate comprehensive beauty analysis report"
    args_schema: Type[BaseModel] = ReportGeneratorInput

    def _run(self, analysis_results: dict):
        try:
            report = {
                "summary": self._generate_summary(analysis_results),
                "personal_analysis": {
                    "face_shape": analysis_results.get("face_shape"),
                    "skin_tone": analysis_results.get("skin_tone"),
                    "personal_color": analysis_results.get("personal_color")
                },
                "recommendations": {
                    "makeup": analysis_results.get("makeup_recommendations"),
                    "hairstyle": analysis_results.get("hairstyle_recommendations")
                },
                "application_guides": {
                    "makeup_steps": self._generate_makeup_steps(analysis_results),
                    "styling_tips": self._generate_styling_tips(analysis_results)
                },
                "product_recommendations": self._generate_product_list(analysis_results),
                "maintenance_schedule": self._generate_maintenance_schedule(analysis_results)
            }
            print ("\nreport :" , report)
            return report
        except Exception as e:
            return {"error": str(e)}

    def _generate_summary(self, analysis_results):
        print ("\nanalysis_results :" , analysis_results)
        return {
            "key_points": [
                "퍼스널 컬러는 가을 웜톤입니다.",
                "둥근 얼굴형에 맞는 헤어스타일을 추천드립니다.",
                "자연스러운 눈매를 강조하는 메이크업을 제안합니다."
            ],
            "overall_concept": "내추럴하면서도 세련된 이미지",
            "focus_points": ["눈매 강조", "광대 쉐이딩", "입체감 있는 헤어스타일"]
        }

    def _generate_maintenance_schedule(self, analysis_results):
        print ("\nanalysis_results :" , analysis_results)
        return {
            "daily": [
                "기초 스킨케어",
                "메이크업 클렌징"
            ],
            "weekly": [
                "딥 클렌징",
                "헤어 트리트먼트"
            ],
            "monthly": [
                "헤어 컷",
                "염색 리터치"
            ]
        }
    
    def analyze_image(self, image_path):
        """
        Analyze uploaded image and generate comprehensive beauty analysis
        """
        try:
            # Initialize analysis results
            analysis_results = {}
            
            # Face detection
            face_detector = FaceDetectionInput(image_path=image_path)
            face_location = self._run(face_detector)
            
            # Facial landmarks detection 
            landmarks_detector = FacialLandmarksTool()
            facial_landmarks = landmarks_detector._run(
                image_path=image_path,
                face_location=face_location
            )
            
            # Extract face shape and features
            analysis_results["face_shape"] = facial_landmarks["face_shape"]
            analysis_results["facial_features"] = facial_landmarks["features"]
            
            # Color analysis
            color_analyzer = ColorAnalysisInput(
                image_path=image_path
            )
            color_analysis = self._run(color_analyzer)
            
            # Determine season and generate color palette
            skin_tone = color_analysis.get("skin_tone", "neutral")
            season_analysis = self._determine_season(skin_tone)
            color_palette = self._generate_color_palette(season_analysis)
            
            analysis_results["skin_tone"] = skin_tone
            analysis_results["personal_color"] = {
                "season": season_analysis,
                "palette": color_palette
            }
            
            # Generate recommendations
            makeup_rec = MakeupRecommendationTool()
            analysis_results["makeup_recommendations"] = makeup_rec._run(
                personal_color=analysis_results["personal_color"],
                face_features=analysis_results["facial_features"]
            )
            print ("\nanalysis_results :" , analysis_results)
            return analysis_results
            
        except Exception as e:
            return {"error": f"Image analysis failed: {str(e)}"}
    
def load_image(image_path: str) -> np.ndarray:
    """이미지를 로드하고 반환합니다."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")
    return image

def detect_face_landmarks(image: np.ndarray) -> dict:
    """OpenCV의 얼굴 검출기를 사용하여 더 상세한 얼굴 특징점을 검출합니다."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        raise Exception("얼굴을 찾을 수 없습니다.")
    
    x, y, w, h = faces[0]
    
    # 더 상세한 얼굴 특징점 생성
    landmarks_dict = {
        # 눈
        "point_left_eye": (x + w//4, y + h//3),
        "point_right_eye": (x + 3*w//4, y + h//3),
        
        # 눈썹
        "point_left_eyebrow": (x + w//4, y + h//4),
        "point_right_eyebrow": (x + 3*w//4, y + h//4),
        
        # 코
        "point_nose_top": (x + w//2, y + 2*h//5),
        "point_nose_bottom": (x + w//2, y + h//2),
        
        # 입
        "point_mouth_left": (x + w//3, y + 2*h//3),
        "point_mouth_right": (x + 2*w//3, y + 2*h//3),
        "point_mouth_top": (x + w//2, y + 5*h//8),
        "point_mouth_bottom": (x + w//2, y + 7*h//10),
        
        # 얼굴 윤곽
        "point_forehead": (x + w//2, y + h//8),
        "point_chin": (x + w//2, y + h),
        "point_jaw_left": (x, y + 3*h//4),
        "point_jaw_right": (x + w, y + 3*h//4),
        "point_cheek_left": (x, y + h//2),
        "point_cheek_right": (x + w, y + h//2),
        "point_temple_left": (x, y + h//4),
        "point_temple_right": (x + w, y + h//4)
    }
    print ("\nlandmarks_dict :" , landmarks_dict)
    return landmarks_dict

def analyze_beauty_features(landmarks: dict, image: np.ndarray) -> Dict[str, Any]:
    """더 상세한 얼굴 분석을 수행합니다."""
    
    face_analysis = {
        "face_shape": {
            "main_shape": determine_face_shape(landmarks),
            "details": analyze_face_shape_details(landmarks)
        },
        "facial_features": {
            "eyes": analyze_eyes(landmarks),
            "nose": analyze_nose(landmarks),
            "lips": analyze_lips(landmarks),
            "jaw": analyze_jaw(landmarks),
            "forehead": analyze_forehead(landmarks),
            "cheeks": analyze_cheeks(landmarks)
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
    return face_analysis

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
    nose_to_chin = calculate_distance(landmarks["point_nose"], landmarks["point_chin"])
    face_height = calculate_distance(landmarks["point_forehead"], landmarks["point_chin"])
    
    nose_analysis = {
        "nose_position_ratio": nose_to_chin / face_height,
        "characteristics": []
    }
    
    if nose_analysis["nose_position_ratio"] > 0.5:
        nose_analysis["characteristics"].append("긴 코")
    elif nose_analysis["nose_position_ratio"] < 0.4:
        nose_analysis["characteristics"].append("짧은 코")
    else:
        nose_analysis["characteristics"].append("표준적인 코 길이")
    print ("\nnose_analysis :" , nose_analysis)
    return nose_analysis

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
    left_eye = np.array(landmarks["point_left_eye"])
    right_eye = np.array(landmarks["point_right_eye"])
    nose = np.array(landmarks["point_nose_bottom"])
    
    left_dist = np.linalg.norm(left_eye - nose)
    right_dist = np.linalg.norm(right_eye - nose)
    
    symmetry = 1 - abs(left_dist - right_dist) / max(left_dist, right_dist)
    print ("\nsymmetry :" , symmetry)
    return float(symmetry)

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
    # 얼굴 비율 계산
    face_width = np.linalg.norm(
        np.array(landmarks["point_cheek_left"]) - 
        np.array(landmarks["point_cheek_right"])
    )
    face_height = np.linalg.norm(
        np.array(landmarks["point_forehead"]) - 
        np.array(landmarks["point_chin"])
    )
    jaw_width = np.linalg.norm(
        np.array(landmarks["point_jaw_left"]) - 
        np.array(landmarks["point_jaw_right"])
    )
    forehead_width = np.linalg.norm(
        np.array(landmarks["point_temple_left"]) - 
        np.array(landmarks["point_temple_right"])
    )
    
    # 비율에 따른 얼굴형 판단
    ratio = face_width / face_height
    jaw_ratio = jaw_width / face_width
    forehead_ratio = forehead_width / face_width
    
    if ratio > 0.9:
        if jaw_ratio < 0.8:
            print ("\n하트형")
            return "하트형"
        else:
            return "둥근형"
    else:
        if jaw_ratio < 0.75:
            if forehead_ratio > 0.9:
                print ("\n역삼각형")
                return "역삼각형"
            else:
                print ("\n계란형")
                return "계란형"
        else:
            if forehead_ratio > 0.9:
                print ("\n사각형")
                return "사각형"
            else:
                print ("\n긴 얼굴형")
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
        return "뾰족한 턱선"
    elif jaw_angle < 100:
        print ("\n부드러운 V라인")
        return "부드러운 V라인"
    else:
        print ("\n둥근 턱선")
        return "둥근 턱선"

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

def analyze_face_proportions(landmarks: dict) -> Dict[str, float]:
    """얼굴 비율을 분석합니다."""
    # 삼정비율 계산 (이마:코:턱)
    forehead_height = np.linalg.norm(
        np.array(landmarks["point_forehead"]) - 
        np.array(landmarks["point_left_eyebrow"])
    )
    midface_height = np.linalg.norm(
        np.array(landmarks["point_left_eyebrow"]) - 
        np.array(landmarks["point_nose_bottom"])
    )
    lower_height = np.linalg.norm(
        np.array(landmarks["point_nose_bottom"]) - 
        np.array(landmarks["point_chin"])
    )
    
    total_height = forehead_height + midface_height + lower_height
    print ("\ntotal_height :" , total_height)
    print ("\nforehead_ratio :" , forehead_height / total_height)
    print ("\nmidface_ratio :" , midface_height / total_height)
    print ("\nlower_ratio :" , lower_height / total_height)
    return {
        "forehead_ratio": forehead_height / total_height,
        "midface_ratio": midface_height / total_height,
        "lower_ratio": lower_height / total_height
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

def save_results(results: Dict[str, Any], output_path: str) -> None:
    """분석 결과를 JSON 파일로 저장합니다."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

class FaceDetection:
    """얼굴 검출을 수행하는 클래스"""
    
    def __init__(self):
        # OpenCV의 얼굴 검출기 초기화
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
    
    def detect_faces(self, image: np.ndarray) -> Dict[str, Any]:
        """
        이미지에서 얼굴을 검출합니다.
        
        Args:
            image: OpenCV 이미지 배열
            
        Returns:
            검출된 얼굴 정보와 랜드마크를 포함한 딕셔너리
        """
        try:
            # 그레이스케일 변환
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 얼굴 검출
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=6,
                minSize=(200, 200)
            )
            
            if len(faces) == 0:
                raise ValueError("이미지에서 얼굴을 찾을 수 없습니다.")
            
            # 가장 큰 얼굴 선택
            main_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = main_face
            
            # 얼굴 영역 추출
            face_region = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(
                face_region,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(30, 30)
            )
            # 검출된 눈 좌표 출력 및 저장
            eye_positions = []
            for (ex, ey, ew, eh) in eyes:
                # 얼굴 영역 내 상대 좌표를 전체 이미지 좌표로 변환
                abs_x = x + ex
                abs_y = y + ey
                print(f"검출된 눈 좌표: x={abs_x}, y={abs_y}, width={ew}, height={eh}")
                eye_positions.append({
                    "x": int(abs_x),
                    "y": int(abs_y),
                    "width": int(ew),
                    "height": int(eh)
                })

            # 랜드마크 생성
            landmarks = self._generate_landmarks(x, y, w, h)
            
            # 얼굴의 상단 부분에서만 눈 검출
            upper_face = face_region[0:int(h*0.5), :]
            
            # 검출된 눈들 중 가장 확실한 두 개만 선택
            def select_best_eyes(eyes):
                # 크기순으로 정렬
                eyes = sorted(eyes, key=lambda x: x[2]*x[3], reverse=True)
                # 상위 2개만 선택
                return eyes[:2]
            
            selected_eyes = select_best_eyes(eyes)
            
            return {
                "eye_position": selected_eyes,
                "face_position": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
                },
                "landmarks": landmarks,
                "eyes_detected": len(selected_eyes),
                "confidence": self._calculate_confidence(gray, x, y, w, h),
                "image_size": {
                    "width": image.shape[1],
                    "height": image.shape[0]
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
            x, y = pos["x"], pos["y"]
            w, h = pos["width"], pos["height"]
            cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # 랜드마크 표시
            for point_name, (px, py) in detection_result["landmarks"].items():
                cv2.circle(vis_image, (px, py), 2, (0, 0, 255), -1)
                cv2.putText(vis_image, point_name, (px, py-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            
            # 신뢰도 표시
            confidence_text = f"Confidence: {detection_result['confidence']:.2f}"
            cv2.putText(vis_image, confidence_text, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 결과 저장
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, vis_image)
            
        except Exception as e:
            raise ValueError(f"시각화 중 오류 발생: {str(e)}")

    def _generate_landmarks(self, x: int, y: int, w: int, h: int) -> dict:
        """얼굴 영역을 기반으로 더 정확한 랜드마크를 생성합니다."""
        return {
            # 이마
            "point_forehead": (
                x + w//2,  # 가로 중앙
                y + int(h * 0.1)  # 상단에서 10% 지점
            ),
            
            # 눈
            "point_left_eye": (
                x + int(w * 0.3),  # 왼쪽에서 30% 지점
                y + int(h * 0.35)  # 상단에서 35% 지점
            ),
            "point_right_eye": (
                x + int(w * 0.7),  # 왼쪽에서 70% 지점
                y + int(h * 0.35)  # 상단에서 35% 지점
            ),
            
            # 코
            "point_nose": (
                x + w//2,  # 가로 중앙
                y + int(h * 0.5)  # 정중앙
            ),
            
            # 입
            "point_mouth_left": (
                x + int(w * 0.35),  # 왼쪽에서 35% 지점
                y + int(h * 0.7)  # 상단에서 70% 지점
            ),
            "point_mouth_right": (
                x + int(w * 0.65),  # 왼쪽에서 65% 지점
                y + int(h * 0.7)  # 상단에서 70% 지점
            ),
            
            # 턱
            "point_chin": (
                x + w//2,  # 가로 중앙
                y + int(h * 0.95)  # 상단에서 95% 지점
            ),
            
            # 볼
            "point_cheek_left": (
                x + int(w * 0.1),  # 왼쪽에서 10% 지점
                y + int(h * 0.5)  # 중앙 높이
            ),
            "point_cheek_right": (
                x + int(w * 0.9),  # 왼쪽에서 90% 지점
                y + int(h * 0.5)  # 중앙 높이
            ),
            
            # 턱선
            "point_jaw_left": (
                x + int(w * 0.15),  # 왼쪽에서 15% 지점
                y + int(h * 0.8)  # 상단에서 80% 지점
            ),
            "point_jaw_right": (
                x + int(w * 0.85),  # 왼쪽에서 85% 지점
                y + int(h * 0.8)  # 상단에서 80% 지점
            ),
            
            # 관자놀이
            "point_temple_left": (
                x + int(w * 0.15),  # 왼쪽에서 15% 지점
                y + int(h * 0.2)  # 상단에서 20% 지점
            ),
            "point_temple_right": (
                x + int(w * 0.85),  # 왼쪽에서 85% 지점
                y + int(h * 0.2)  # 상단에서 20% 지점
            ),
            
            # 눈썹
            "point_eyebrow_left": (
                x + int(w * 0.3),  # 왼쪽에서 30% 지점
                y + int(h * 0.25)  # 상단에서 25% 지점
            ),
            "point_eyebrow_right": (
                x + int(w * 0.7),  # 왼쪽에서 70% 지점
                y + int(h * 0.25)  # 상단에서 25% 지점
            )
        }
    
    def _calculate_confidence(self, gray_image: np.ndarray, x: int, y: int, w: int, h: int) -> float:
        """얼굴 검출 신뢰도를 계산합니다."""
        face_region = gray_image[y:y+h, x:x+w]
        mean_intensity = np.mean(face_region)
        std_intensity = np.std(face_region)
        return float(min(1.0, (std_intensity / mean_intensity) * 0.5))

class ImageAnalysisResponse(BaseModel):
    """이미지 분석 결과 응답 모델"""
    face_detection: Dict[str, Any]
    face_characteristics: Dict[str, Any]
    measurements: Dict[str, Any]
    skin_analysis: Dict[str, Any]

# 전역 인스턴스 생성
face_detector = FaceDetection()

@app.post("/analyze", response_model=ImageAnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    """이미지를 업로드받아 얼굴 분석을 수행합니다."""
    try:
        # 얼굴 검출
        face_detection_result = await face_detector.detect_faces(file)
        
        # 특징 분석 수행
        landmarks = face_detection_result["landmarks"]
        
        # 파일 다시 읽기 (분석용)
        contents = await file.seek(0)
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        analysis_result = analyze_beauty_features(landmarks, image)
        
        # 전체 결과 구성
        result = {
            "face_detection": face_detection_result,
            "face_characteristics": analysis_result["face_characteristics"],
            "measurements": analysis_result["measurements"],
            "skin_analysis": analysis_result["skin_analysis"]
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def main():
    try:
        print("얼굴 분석 프로그램을 시작합니다...")
        
        # 이미지 파일 브라우저로 선택
        print("이미지 파일을 선택해주세요...")
        image_path = browse_image()
        print(f"선택된 이미지: {image_path}")
        
        # 얼굴 검출기 인스턴스 생성
        detector = FaceDetection()
        
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
        
        # 얼굴 검출 수행
        result = detector.detect_faces(image)
        print("검출 결과:", result)
        
        # 결과 저장 경로 생성
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"detected_{os.path.basename(image_path)}")
        
        # 결과 시각화
        detector.visualize_detection(image, result, output_path)
        print(f"결과 이미지가 저장되었습니다: {output_path}")
        
        # 결과를 메시지 박스로 표시
        root = tk.Tk()
        root.withdraw()
        tk.messagebox.showinfo("분석 완료", 
            f"얼굴 분석이 완료되었습니다.\n\n"
            f"검출된 얼굴 신뢰도: {result['confidence']:.2f}\n"
            f"검출된 눈 개수: {result['eyes_detected']}\n\n"
            f"결과 이미지가 저장된 경로:\n{output_path}"
        )
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        if 'root' in locals():
            tk.messagebox.showerror("오류", f"분석 중 오류가 발생했습니다:\n{str(e)}")

if __name__ == "__main__":
    main()

def analyze_lips(landmarks: dict) -> Dict[str, Any]:
    lip_width = calculate_distance(landmarks["point_mouth_left"], landmarks["point_mouth_right"])
    face_width = calculate_distance(landmarks["point_cheek_left"], landmarks["point_cheek_right"])
    return {"lip_width_ratio": lip_width / face_width}

def analyze_jaw(landmarks: dict) -> Dict[str, Any]:
    jaw_width = calculate_distance(landmarks["point_jaw_left"], landmarks["point_jaw_right"])
    face_width = calculate_distance(landmarks["point_cheek_left"], landmarks["point_cheek_right"])
    return {"jaw_ratio": jaw_width / face_width}

def analyze_forehead(landmarks: dict) -> Dict[str, Any]:
    forehead_width = calculate_distance(landmarks["point_temple_left"], landmarks["point_temple_right"])
    face_width = calculate_distance(landmarks["point_cheek_left"], landmarks["point_cheek_right"])
    return {"forehead_ratio": forehead_width / face_width}

