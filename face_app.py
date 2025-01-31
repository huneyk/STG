import os
import warningfilter
# warnings.filterwarnings('ignore')
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai.process import Process
from crewai_tools import SerperDevTool, WebsiteSearchTool, ScrapeWebsiteTool
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

serper_api_key = os.getenv("SERPER_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# 2️⃣ CrewAI 및 관련 도구 임포트
# CrewAI 핵심 컴포넌트 임포트
from crewai import Agent, Task, Crew
from crewai.process import Process

# CrewAI 도구 임포트
from crewai_tools import (
    SerperDevTool,
    WebsiteSearchTool,
    ScrapeWebsiteTool
)

# OpenAI LLM 설정
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")

# 3️⃣ 검색 도구 초기화
# 웹 검색 및 스크래핑을 위한 도구 설정
search_tool = SerperDevTool()
web_rag_tool = WebsiteSearchTool()
scrap_tool = ScrapeWebsiteTool()

# 4️⃣ 사용자 입력 받기
# 주요 필드와 토픽 입력
# File upload functionality
import tkinter as tk
from tkinter import filedialog

def browse_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp"),
            ("All files", "*.*")
        ]
    )
    
    if file_path:
        print(f"Selected file: {file_path}")
        return file_path
    else:
        print("No file selected")
        return None

# Get image file path
image_path = browse_file()


# field = input("Enter the main field of expertise: ")
# topic = input("Please specify topics of the field: ")

###
#  톤 선택
''' tone = ["funny", "informative", "Emotional"]
num = 1
for i in tone:
    print(num, " : ", i)
    num += 1

input_tone = int(input("Enter the number of tone you want: "))
select_tone = tone[input_tone-1]
'''
###

# 결과 파일명 입력
file_name = input("Enter the file name for the result: ")

"""Agent 정의

"""
# Import required libraries for image handling
from PIL import Image
import base64
from io import BytesIO

def encode_image_to_base64(image_path):
    """Convert image to base64 string"""
    try:
        # Open and encode image
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Create byte buffer
            buffered = BytesIO()
            # Save image to buffer
            img.save(buffered, format="JPEG")
            # Encode to base64
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return img_str
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

# Initialize ChatOpenAI with the image
if image_path:
    image_base64 = encode_image_to_base64(image_path)
    if image_base64:
        # Create a new ChatOpenAI instance with the image in the system message
        llm = ChatOpenAI(
            model="gpt-4-vision-preview",  # Make sure to use vision model
            max_tokens=4096,
            temperature=0.7,
            messages=[{
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant that can see and analyze images."
                    },
                    {
                        "type": "image",
                        "image_url": f"data:image/jpeg;base64,{image_base64}"
                    }
                ]
            }]
        )
        print("Image successfully added to LLM context")
    else:
        print("Failed to encode image")

# 5️⃣ Researcher 에이전트 정의
researcher = Agent(
    role="Professional Researcher about " + field,
    goal= field + topic +'에 대한 포괄적인 리서치를 수행하고 트렌드, 경쟁사, 시청자 관심사를 분석합니다',
    backstory="""15년 차 디지털 마케팅 전문가로, 소셜 미디어 트렌드 분석과 콘텐츠 전략 수립을 전문으로 합니다. Google Analytics와 YouTube Analytics 전문가 자격을 보유하고 있으며, 빅데이터 분석을 통해 시청자들의 관심사와 행동 패턴을 정확하게 파악합니다""",
    tools=[search_tool, web_rag_tool],
    verbose=True,
    max_iter=10,
    llm=llm
)

# 6️⃣ Creator 에이전트 정의
creator = Agent(
    role="Professional YouTube Creator",
    goal='리서치 결과를 바탕으로 스토리보드, 내러티브, CTA를 포함해' + select_tone + '의 tone으로 매력적인 콘텐츠 계획을 수립합니다',
    backstory="""당신은 바이럴 유튜브 콘텐츠 제작 분야에서 10년의 경력을 가진 크리에이티브 디렉터입니다.
    영화학과 심리학을 전공하여 감정을 자극하는 스토리텔링 분야의 전문가입니다.유튜브에서 백만뷰 이상을 기록한 동영상을 만든 적이 있습니다.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

writer = Agent(
    role='YouTube 시나리오 작가',
    goal="""리서치 결과와 콘텐츠 계획을 바탕으로 매력적인 시나리오를 작성하고,
    시청자의 관심을 끌고 유지하는 효과적인 대사를 구성하며,
    브랜드/채널의 톤앤매너에 맞는 일관된 어투를 개발합니다.
    시청자 층에 맞는 적절한 언어를 사용하고,
    명확하고 실행 가능한 제작 지침을 제공합니다.""",
    backstory="""당신은 YouTube 콘텐츠 제작에 특화된 10년 경력의 전문 시나리오 작가입니다.
    100만 이상의 조회수를 기록한 다수의 바이럴 콘텐츠 제작에 참여했으며,
    교육, 엔터테인먼트, 정보성 콘텐츠 등 다양한 장르의 크리에이터들과 협업한 경험이 있습니다.

    시청자 심리학과 온라인 콘텐츠 소비 패턴에 대한 깊은 이해를 바탕으로,
    시청자의 관심을 사로잡고 유지하는 스토리텔링에 탁월한 능력을 보유하고 있습니다.

    특히 다음과 같은 분야에서 전문성을 인정받고 있습니다:
    - 시청자를 사로잡는 매력적인 도입부 작성
    - 핵심 메시지의 효과적인 전달
    - 자연스러운 에피소드 전환
    - 시청자 참여를 유도하는 종결부 구성
    - 채널/브랜드 톤앤매너의 일관성 유지""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    memory=True
)

print("Agents defined successfully! \n")


"""Task 정의"""

# 7️⃣ 태스크 정의
# Research 태스크
research = Task(
    description=f"""주제 {field}과 {topic}에 대해 다음 항목들을 철저히 리서치하세요:
        1. 현재 YouTube 트렌드 분석
        2. 성과가 좋은 콘텐츠 파악
        3. 핵심 시청자층 분석
        4. 성공적인 영상 포맷과 스타일 분석
        5. SEO 키워드 도출
        """,
    agent=researcher,
    context=[
        {
            "description": f"Research about {field}",
            "expected_output": "Detailed analysis of the field"
        },
        {
            "description": f"Research about {topic}",
            "expected_output": "Detailed analysis of the topic"
        },
        {
            "description": f"Apply {tone} tone",
            "expected_output": "Content with appropriate tone"
        }
    ],
    expected_output="""다음 섹션들을 포함한 상세한 리서치 보고서를 작성해주세요:

1. 트렌드 분석
   - 현재 유튜브에서 인기 있는 주요 트렌드 설명
   - 성공적인 영상의 포맷과 스타일 분석
   - 시청자들의 주요 선호도와 시청 패턴
   - 향후 트렌드 예측

2. 경쟁 채널 분석
   - 주요 경쟁 채널들의 특징과 성과
   - 구독자 수와 평균 조회수 데이터
   - 성공적인 영상들의 공통된 특징
   - 차별화 가능한 틈새 시장 기회

3. 시청자 분석
   - 주요 타겟 시청자층의 특성
   - 연령대별 선호 콘텐츠 유형
   - 주요 관심사와 니즈
   - 시청자 참여도가 높은 콘텐츠 특징

4. SEO 전략
   - 상위 노출이 기대되는 주요 키워드
   - 보조 키워드 및 연관 검색어
   - 효과적인 태그 조합 제안
   - 검색 최적화를 위한 제목 구성 전략

5. 제작 전략 제안
   - 리서치 결과를 바탕으로 한 차별화 전략
   - 조회수와 구독자 증가를 위한 실행 방안
   - 예상되는 위험 요소와 대응 방안
   - 중장기 콘텐츠 방향성 제안

각 섹션은 구체적인 데이터와 예시를 포함하여 작성해주세요. 또한 중요한 인사이트는 굵게 표시하여 강조해주세요."""
    )


# Create 태스크
create = Task(
    description=f"""리서치 결과를 바탕으로 주제 {field}에 대해 {topic}에 중점을 두어서 상세한 콘텐츠 계획을 수립하세요:
        1. 매력적인 스토리보드 디자인
        2. 감정을 자극하는 내러티브 구조 개발
        3. 전략적 CTA 배치
        4. 시각적 요소와 전환 계획
        5. 썸네일 컨셉 구상
        6. 동영상 전체 화면별 대사
        """,
    agent=creator,
    context=[{
        "description": f"Topic: {field}, Focus: {topic}",
        "expected_output": "Detailed content plan"
    }],
    expected_output="""다음 섹션들을 포함한 상세한 콘텐츠 계획을 작성해주세요:

1. 스토리보드
   - 썸네일 디자인
     * 핵심 시각적 요소 설명
     * 사용할 텍스트/문구 제안
     * 시청자의 클릭을 유도하는 차별화 포인트

   - 영상 구조
     * 섹션별 상세 구성안
     * 각 섹션의 예상 길이
     * 주요 시각적 요소와 화면 구성
     * 섹션 간 전환 방식

2. 내러티브 구조
   - 도입부
     * 시청자의 관심을 사로잡을 오프닝 문구
     * 핵심 메시지 제시 방법
     * 시청자 기대감 형성 전략

   - 본문 구성
     * 섹션별 핵심 메시지
     * 스토리텔링 전개 방식
     * 감정적 요소와 몰입도 향상 전략

   - 결론
     * 핵심 내용 정리 방식
     * 인상적인 마무리 문구
     * 다음 콘텐츠로의 연결 요소

3. CTA(시청자 참여 유도) 전략
   - 시간대별 CTA 배치 계획
     * 주요 CTA 문구와 배치 시점
     * 자연스러운 유도 방식
     * 예상 반응과 대응 방안

   - 시청자 참여 유도 전략
     * 구독과 좋아요 유도 방식
     * 댓글 참여 활성화 전략
     * 커뮤니티 참여 유도 방안

4. 제작 가이드라인
   - 기술적 스펙
     * 영상 길이 및 포맷
     * 화질과 음질 요구사항
     * 자막 및 텍스트 스타일

   - 필요 리소스
     * 촬영 필요 장면 목록
     * 그래픽 요소 명세
     * 배경음악 및 효과음 가이드
     * 기타 필요 자료

각 섹션은 구체적이고 실행 가능한 내용으로 작성해주세요. 특히 중요한 차별화 요소나 핵심 포인트는 굵게 표시하여 강조해주시기 바랍니다. 또한 리서치 결과에서 도출된 인사이트가 각 섹션에 어떻게 반영되었는지 설명해주세요.""",


    output_file=r"idea_plan_"+file_name+".md"
)


# Create script writing task
script = Task(
    description=f"""리서치 결과와 콘텐츠 계획을 바탕으로 주제 '{topic}'에 대한 상세 시나리오를 작성하세요:
        1. 각 장면별 구체적인 대사와 지문 작성
        2. 내레이션과 강조점 표시
        3. 영상 특수효과와 자막 표시 위치 지정
        4. 배경음악과 효과음 타이밍 명시
        5. 촬영 가이드라인 포함
        """,
    agent=writer,
    context=[{
        "description": "research 결과와 create 결과를 바탕으로 주제 {topic}에 대한 상세 시나리오를 작성하세요",  # Add this
        "expected_output": "창의적이고 매력적인 시나리오를 작성하세요",  # Add this
        "field": field,
        "topics": topic,
        "selected_tone": select_tone
    }],
    expected_output="""다음 형식으로 상세 시나리오를 작성해주세요:

1. 영상 기본 정보
   - 예상 러닝타임
   - 메인 타깃 시청자
   - 전체적인 톤앤매너
   - 핵심 전달 메시지

2. 장면별 상세 시나리오
   [인트로]
   - 영상 구성 요소: (화면 설명)
   - 배경음악: (음악 설명)
   - 대사/내레이션: "구체적인 대사 내용"
   - 자막/효과: [자막 표시 내용]
   - 특수효과: (효과 설명)
   - 촬영 가이드: (카메라 앵글, 구도 등)

   [메인 내용 - 섹션별로 위 형식을 반복]
   - 섹션 1
   - 섹션 2
   ...

   [아웃트로]
   - CTA 및 엔딩 카드 구성

3. 제작 참고사항
   - 특별히 신경써야 할 장면이나 대사
   - 편집 시 주의사항
   - 음향 효과 가이드라인
   - 자막 스타일 가이드

각 장면은 최대한 구체적으로 작성해주세요. 특히:
- 대사는 실제 발화할 내용을 정확히 작성
- 화면 구성은 시청자가 볼 내용을 자세히 설명
- 효과와 자막은 시간 타이밍을 포함하여 명시
- 촬영 가이드는 실제 제작에 필요한 디테일을 포함

또한 시청자의 몰입도를 높이기 위한 요소들을 적절히 배치해주세요.""",


    output_file=r"idea_script_"+file_name+".md"
)

print("Tasks defined successfully! \n")

# 8️⃣ Crew 설정 및 실행
# Crew 객체 생성 및 실행
crew = Crew(
    agents=[researcher, creator, writer],
    tasks=[research, create, script],
    verbose=True,
    process=Process.sequential
)

# 결과 실행
result = crew.kickoff()
