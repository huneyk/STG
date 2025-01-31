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
            return report
        except Exception as e:
            return {"error": str(e)}

    def _generate_summary(self, analysis_results):
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
    """OpenCV의 기본 얼굴 검출기를 사용합니다."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 검출
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        raise Exception("얼굴을 찾을 수 없습니다.")
    
    # 첫 번째 얼굴 사용
    x, y, w, h = faces[0]
    
    # 간단한 랜드마크 생성 (얼굴 사각형의 주요 지점들)
    landmarks_dict = {
        "point_left_eye": (x + w//4, y + h//3),
        "point_right_eye": (x + 3*w//4, y + h//3),
        "point_nose": (x + w//2, y + h//2),
        "point_mouth": (x + w//2, y + 2*h//3),
        "point_chin": (x + w//2, y + h),
        "point_left": (x, y + h//2),
        "point_right": (x + w, y + h//2),
        "point_top": (x + w//2, y)
    }
    
    return landmarks_dict

def analyze_beauty_features(landmarks: dict) -> Dict[str, Any]:
    """단순화된 미용 분석을 수행합니다."""
    results = {
        "face_symmetry": calculate_symmetry(landmarks),
        "face_ratio": calculate_face_ratio(landmarks),
    }
    return results

def calculate_symmetry(landmarks: dict) -> float:
    """얼굴의 대칭성을 계산합니다."""
    left_eye = np.array(landmarks["point_left_eye"])
    right_eye = np.array(landmarks["point_right_eye"])
    nose = np.array(landmarks["point_nose"])
    
    left_dist = np.linalg.norm(left_eye - nose)
    right_dist = np.linalg.norm(right_eye - nose)
    
    symmetry = 1 - abs(left_dist - right_dist) / max(left_dist, right_dist)
    return float(symmetry)

def calculate_face_ratio(landmarks: dict) -> float:
    """얼굴의 가로 세로 비율을 계산합니다."""
    left = np.array(landmarks["point_left"])
    right = np.array(landmarks["point_right"])
    top = np.array(landmarks["point_top"])
    chin = np.array(landmarks["point_chin"])
    
    width = np.linalg.norm(right - left)
    height = np.linalg.norm(chin - top)
    
    return float(width / height) if height != 0 else 0

def browse_image() -> str:
    """파일 브라우저를 열어 이미지 파일을 선택합니다."""
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="분석할 이미지 선택",
        filetypes=[
            ("이미지 파일", "*.jpg *.jpeg *.png *.bmp"),
            ("모든 파일", "*.*")
        ]
    )
    
    if not file_path:
        raise Exception("이미지가 선택되지 않았습니다.")
    
    return file_path

def save_results(results: Dict[str, Any], output_path: str) -> None:
    """분석 결과를 JSON 파일로 저장합니다."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

def main():
    try:
        print("얼굴 분석 프로그램을 시작합니다...")
        
        # 이미지 파일 브라우저로 선택
        print("이미지 파일을 선택해주세요...")
        image_path = browse_image()
        print(f"선택된 이미지: {image_path}")
        
        # 이미지 로드
        image = load_image(image_path)
        
        # 결과 저장 경로 생성
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "result.json")
        
        # 얼굴 검출 및 특징점 추출
        face_landmarks = detect_face_landmarks(image)
        
        # 미용 분석 수행
        analysis_result = analyze_beauty_features(face_landmarks)
        
        # 결과 저장
        save_results(analysis_result, output_path)
        
        print("분석이 성공적으로 완료되었습니다.")
        print(f"결과가 저장된 경로: {output_path}")
        
        # 결과를 메시지 박스로 표시
        messagebox.showinfo("분석 완료", 
            f"분석이 완료되었습니다.\n"
            f"대칭성: {analysis_result['face_symmetry']:.2f}\n"
            f"얼굴 비율: {analysis_result['face_ratio']:.2f}"
        )
        
        return True
        
    except FileNotFoundError:
        print("이미지 파일을 찾을 수 없습니다.")
        messagebox.showerror("오류", "이미지 파일을 찾을 수 없습니다.")
        return False
    except Exception as e:
        print(f"분석 중 오류가 발생했습니다: {str(e)}")
        messagebox.showerror("오류", f"분석 중 오류가 발생했습니다:\n{str(e)}")
        return False

if __name__ == "__main__":
    main()

