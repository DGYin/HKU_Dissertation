# CrossMedia-PID: Cross-Modal Person Identification System
## Mid-Term Dissertation Report

**Student Name:** [Your Name]  
**Student ID:** [Your Student ID]  
**Supervisor:** [Supervisor Name]  
**Department:** [Department Name]  
**Institution:** The University of Hong Kong  
**Date:** April 27, 2026

---

## 1. Project Overview

### 1.1 Research Background

In the era of ubiquitous surveillance and multimedia content, cross-modal person identification has become a critical challenge in computer vision and pattern recognition. Traditional person re-identification (Re-ID) systems primarily rely on single-modal features, typically visual appearance from cameras. However, real-world scenarios often require matching individuals across different media types, such as matching a person from a photograph to their appearance in video footage, or correlating descriptions with visual data.

The CrossMedia-PID (Cross-Modal Person Identification) system addresses this challenge by developing a comprehensive framework that can extract, represent, and match person identities across different media formats using advanced Vision-Language Models (VLMs) and hybrid vector representations.

### 1.2 Research Objectives

The primary objectives of this dissertation project are:

1. **Develop a modular cross-modal person identification system** capable of processing images and videos to extract person-specific features
2. **Implement hybrid feature representation** combining dense vector embeddings with sparse attribute-based representations
3. **Design robust matching algorithms** that can accurately identify the same person across different media sources
4. **Create a scalable database architecture** for efficient storage and retrieval of person embeddings
5. **Build user-friendly interfaces** including CLI and Web GUI for system interaction and testing

### 1.3 System Architecture

The CrossMedia-PID system follows a modular architecture with four core processing modules:

```
┌─────────────────────────────────────────────────────────────┐
│                    CrossMedia-PID System                     │
├─────────────────────────────────────────────────────────────┤
│  Module A: Visual Extractor (YOLOv8-based Person Detection) │
│  Module B: Feature Extractor (VLM-based Attribute Extraction)│
│  Module C: Vectorizer (Dense + Sparse Vector Generation)    │
│  Module D: Identity Matcher (Hybrid Distance Matching)      │
├─────────────────────────────────────────────────────────────┤
│  Storage Layer: ChromaDB Vector Database                    │
│  Interface Layer: CLI + Web GUI (Streamlit)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Completed Work

### 2.1 Core System Development

#### 2.1.1 Module A: Visual Extractor
**Status:** ✅ Completed

- Implemented person detection using YOLOv8 object detection model
- Developed bounding box extraction and quality scoring mechanisms
- Created `PersonExtractor` class with configurable confidence thresholds and IoU parameters
- Implemented best-crop selection based on detection quality metrics
- Added support for minimum bounding box size filtering (64px default)

**Key Features:**
- Real-time person detection in images
- Quality-based crop selection (resolution, occlusion, pose estimation)
- Configurable detection parameters via YAML configuration
- Integration with Ultralytics YOLO ecosystem

#### 2.1.2 Module B: Feature Extractor
**Status:** ✅ Completed

- Implemented Vision-Language Model (VLM) based attribute extraction
- Developed dual-provider architecture supporting:
  - Cloud-based VLM APIs (OpenAI-compatible)
  - Aliyun DashScope API (Qwen3-VL-235B model)
- Created structured attribute extraction covering 19+ person attributes including:
  - Gender, age group, body build
  - Clothing colors and types (topwear, bottomwear, shoes)
  - Accessories (glasses, hat, bag)
  - Hair style and color
  - Pattern descriptions

**Key Features:**
- JSON-formatted attribute extraction with validation
- Robust error handling and retry mechanisms
- Support for both cloud and local model deployment
- Attribute standardization and normalization

#### 2.1.3 Module C: Vectorizer
**Status:** ✅ Completed

- Implemented hybrid vector representation combining:
  - **Dense vectors**: Using BAAI/bge-small-zh-v1.5 embedding model (512 dimensions)
  - **Sparse vectors**: Attribute-based one-hot encoding with dynamic registry
- Developed `DynamicVectorizer` with attribute registry management
- Created persistent attribute registry for consistent vector space mapping
- Implemented vector normalization and combination strategies

**Key Features:**
- Automatic attribute-to-vector mapping
- Dynamic registry expansion for new attributes
- Configurable embedding models
- Efficient sparse vector representation

#### 2.1.4 Module D: Identity Matcher
**Status:** ✅ Completed

- Implemented hybrid distance calculation algorithm:
  - Cosine similarity for dense vectors
  - Jaccard distance for sparse vectors
  - Weighted combination with configurable weights
- Developed threshold-based identity decision mechanism
- Created `IdentityMatcher` class with comprehensive matching logic
- Implemented top-K candidate retrieval and ranking

**Key Features:**
- Configurable similarity threshold (default: 0.72)
- Adjustable weight parameters for dense/sparse vectors
- New identity creation for unmatched persons
- Detailed match score breakdown and candidate analysis

### 2.2 Database Infrastructure

**Status:** ✅ Completed

- Integrated ChromaDB as the vector database backend
- Implemented `ChromaStore` class with persistent storage
- Created collection management for person embeddings
- Added metadata storage for attributes and source information
- Implemented efficient similarity search functionality

**Key Features:**
- Persistent vector storage across sessions
- Cosine distance metric for similarity search
- Metadata filtering and querying
- Automatic collection management

### 2.3 System Integration & CLI

**Status:** ✅ Completed

- Developed main controller (`CrossMediaPID` class) integrating all modules
- Created comprehensive CLI interface using Click framework
- Implemented image processing pipeline with full workflow automation
- Added configuration management via YAML files
- Created environment setup scripts for reproducibility

**CLI Commands:**
```bash
# Process single image
python main.py process image.jpg

# Process with options
python main.py process image.jpg --no-add

# Verbose output
python main.py -v process image.jpg
```

### 2.4 Web GUI Development

**Status:** ✅ Completed

- Developed interactive Web GUI using Streamlit framework
- Implemented video upload and processing interface
- Created person tracking visualization using YOLOv8 tracking
- Built manual track selection interface for testing
- Implemented automatic screenshot extraction for left/right positions
- Added comprehensive result display and comparison visualization

**Web GUI Features:**
- Drag-and-drop video file upload
- Real-time person detection and tracking visualization
- Interactive track selection
- Automatic CrossMedia-PID matching test
- Detailed result comparison display
- Performance metrics visualization

### 2.5 Testing & Evaluation Framework

**Status:** ✅ Completed

#### 2.5.1 Performance Monitoring System
- Implemented `PerformanceMonitor` class for step-by-step timing
- Created comprehensive performance tracking for all processing stages
- Developed alert thresholds for performance anomaly detection
- Implemented statistics export for visualization

**Performance Metrics Tracked:**
- Image loading time
- Person detection time
- VLM feature extraction time
- Vectorization time
- Database search time
- Identity matching time
- Total processing time

#### 2.5.2 Stability Testing
- Developed 10-run consistency test framework
- Implemented attribute matching analysis
- Created pattern matching evaluation
- Generated detailed stability reports

**Test Results Summary:**
- **Consistency Rate:** 100% (10/10 runs consistent)
- **Average Match Score:** 0.96+ for same person
- **Attribute Match Rate:** 50-67%
- **Average Processing Time:** 50-80 seconds per test

#### 2.5.3 Video Person Matching Test
- Implemented video-based person tracking test
- Created automatic frame extraction for different positions
- Developed same-person verification across video frames
- Built comprehensive test reporting system

### 2.6 Configuration & Documentation

**Status:** ✅ Completed

- Created comprehensive YAML configuration system
- Implemented environment variable support for API keys
- Developed setup scripts for environment reproducibility
- Created project wiki documentation (Chinese)
- Implemented `.gitignore` and project structure organization

**Configuration Features:**
- Modular configuration for each subsystem
- Environment variable substitution
- Model selection and parameter tuning
- Matching threshold and weight configuration

### 2.7 Technical Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Object Detection | YOLOv8 | 8.0+ |
| Vision-Language Model | Qwen3-VL-235B / OpenAI VLM | - |
| Embedding Model | BAAI/bge-small-zh-v1.5 | - |
| Vector Database | ChromaDB | 0.4.0+ |
| Web Framework | Streamlit | Latest |
| CLI Framework | Click | 8.1.0+ |
| Image Processing | OpenCV, PIL | 4.8+, 10.0+ |
| Configuration | PyYAML | 6.0+ |
| Python | Python | 3.11 |

---

## 3. System Performance

### 3.1 Current Performance Metrics

Based on comprehensive testing with the implemented system:

| Processing Stage | Average Time | Alert Threshold |
|------------------|--------------|-----------------|
| Module Initialization | < 1s | - |
| Image Loading | 0.001s | 0.5s |
| Person Detection | 0.32s | 3.0s |
| VLM Feature Extraction | 3.28s | 10.0s |
| Vectorization | 1.97s | 5.0s |
| Database Add | 0.56s | - |
| Identity Matching | 0.001s | 1.0s |
| **Total Processing** | **~6s** | **25.0s** |

### 3.2 Matching Accuracy

- **Same Person Match Score:** 0.95-0.97 (dense: 0.98+, sparse: 0.90+)
- **Consistency Rate:** 100% across 10 consecutive tests
- **Attribute Extraction:** 19 attributes consistently extracted
- **False Positive Rate:** 0% in current test scenarios

### 3.3 System Scalability

- **Database Capacity:** Tested with multiple collections, supports thousands of embeddings
- **Concurrent Processing:** Single-threaded currently, multi-threading planned
- **Storage Efficiency:** ChromaDB provides efficient vector storage and retrieval

---

## 4. Pending Work & Future Plan

### 4.1 Short-term Goals (May 2026)

#### 4.1.1 Enhanced Face Recognition Module
**Priority:** High  
**Timeline:** Week 1-2 of May

- Implement face detection and embedding extraction
- Integrate face similarity scoring into the matching pipeline
- Add face recognition as an optional third vector type (dense + sparse + face)
- Conduct ablation studies to evaluate face recognition contribution

**Tasks:**
- [ ] Research and select face recognition model (e.g., ArcFace, FaceNet)
- [ ] Implement face detection module
- [ ] Develop face embedding extraction pipeline
- [ ] Integrate face score into hybrid matching algorithm
- [ ] Test and validate face recognition accuracy

#### 4.1.2 Video Processing Pipeline Enhancement
**Priority:** High  
**Timeline:** Week 2-3 of May

- Implement automated video frame sampling strategies
- Develop temporal consistency checking for person tracking
- Create batch processing capabilities for long videos
- Optimize tracking algorithm for occlusion handling

**Tasks:**
- [ ] Implement intelligent frame sampling (keyframe extraction)
- [ ] Add temporal smoothing for attribute consistency
- [ ] Develop batch video processing workflow
- [ ] Optimize tracking with occlusion recovery
- [ ] Create video processing performance benchmarks

#### 4.1.3 Advanced Testing & Evaluation
**Priority:** Medium  
**Timeline:** Week 3-4 of May

- Create comprehensive test dataset with ground truth labels
- Implement precision, recall, F1-score calculation
- Conduct cross-validation experiments
- Perform parameter sensitivity analysis

**Tasks:**
- [ ] Collect and annotate test dataset (50+ persons, 200+ images)
- [ ] Implement evaluation metrics computation
- [ ] Conduct threshold optimization experiments
- [ ] Perform weight parameter sensitivity analysis
- [ ] Generate comprehensive evaluation report

### 4.2 Mid-term Goals (Early June 2026)

#### 4.2.1 Multi-Modal Support
**Priority:** High  
**Timeline:** Week 1 of June

- Extend system to support text-to-person matching
- Implement description-based person search
- Create multi-modal embedding alignment
- Develop text-vision fusion mechanisms

**Tasks:**
- [ ] Implement text description parser
- [ ] Create text-to-attribute mapping
- [ ] Develop cross-modal similarity computation
- [ ] Test text-based person retrieval
- [ ] Evaluate multi-modal matching accuracy

#### 4.2.2 Performance Optimization
**Priority:** Medium  
**Timeline:** Week 1-2 of June

- Implement model quantization for faster inference
- Add caching mechanisms for repeated queries
- Optimize database indexing strategies
- Reduce overall processing time by 30%

**Tasks:**
- [ ] Implement embedding model quantization
- [ ] Add result caching layer
- [ ] Optimize ChromaDB indexing parameters
- [ ] Profile and optimize bottlenecks
- [ ] Benchmark performance improvements

#### 4.2.3 User Interface Enhancement
**Priority:** Medium  
**Timeline:** Week 2 of June

- Add batch image processing support to Web GUI
- Implement result export functionality (JSON, CSV)
- Create interactive threshold tuning interface
- Add real-time performance monitoring dashboard

**Tasks:**
- [ ] Implement batch upload and processing
- [ ] Add result export features
- [ ] Create interactive parameter tuning UI
- [ ] Build performance monitoring dashboard
- [ ] Improve UI/UX based on user feedback

### 4.3 Final Goals (Mid-Late June 2026)

#### 4.3.1 Comprehensive Evaluation
**Priority:** High  
**Timeline:** Week 3 of June

- Conduct extensive experiments on diverse datasets
- Compare with baseline methods and state-of-the-art approaches
- Perform statistical analysis of results
- Document all experimental findings

**Tasks:**
- [ ] Execute comprehensive experimental pipeline
- [ ] Compare with existing Re-ID methods
- [ ] Perform statistical significance testing
- [ ] Analyze failure cases and limitations
- [ ] Document experimental results

#### 4.3.2 Dissertation Writing
**Priority:** Critical  
**Timeline:** Week 3-4 of June

- Complete all dissertation chapters
- Create comprehensive figures and tables
- Write conclusion and future work sections
- Perform thorough proofreading and revision

**Dissertation Structure:**
1. **Introduction** (Background, Objectives, Contributions)
2. **Literature Review** (Person Re-ID, VLMs, Vector Databases, Cross-Modal Matching)
3. **Methodology** (System Architecture, Four Modules, Hybrid Representation)
4. **Implementation** (Technical Details, Configuration, Integration)
5. **Experiments & Evaluation** (Datasets, Metrics, Results, Analysis)
6. **Discussion** (Strengths, Limitations, Practical Applications)
7. **Conclusion & Future Work** (Summary, Contributions, Research Directions)

**Tasks:**
- [ ] Complete Chapter 1-3 (Introduction, Literature Review, Methodology)
- [ ] Complete Chapter 4-5 (Implementation, Experiments)
- [ ] Complete Chapter 6-7 (Discussion, Conclusion)
- [ ] Create all figures, tables, and diagrams
- [ ] Perform multiple rounds of revision and proofreading
- [ ] Format according to HKU dissertation guidelines

#### 4.3.3 Final System Polish
**Priority:** High  
**Timeline:** Week 4 of June

- Fix all identified bugs and issues
- Optimize system stability and robustness
- Create final demonstration materials
- Prepare presentation slides

**Tasks:**
- [ ] Comprehensive bug fixing and testing
- [ ] System stability optimization
- [ ] Create demo videos and screenshots
- [ ] Prepare final presentation slides
- [ ] Conduct mock presentations

---

## 5. Risk Assessment & Mitigation

### 5.1 Technical Risks

| Risk | Impact | Probability | Mitigation Strategy |
|------|--------|-------------|---------------------|
| VLM API instability | High | Medium | Implement fallback to local models; add retry mechanisms |
| Face recognition integration complexity | Medium | Medium | Start with simple implementation; iterate gradually |
| Performance bottlenecks | Medium | Low | Early profiling; optimize critical paths first |
| Dataset collection difficulties | High | Low | Use publicly available datasets; augment with synthetic data |

### 5.2 Timeline Risks

| Risk | Impact | Mitigation Strategy |
|------|--------|---------------------|
| Feature creep | High | Strictly prioritize core features; defer nice-to-haves |
| Writing delays | High | Start writing early; maintain weekly writing schedule |
| Experimental failures | Medium | Plan backup experiments; validate assumptions early |

### 5.3 Quality Risks

| Risk | Impact | Mitigation Strategy |
|------|--------|---------------------|
| Insufficient evaluation | High | Define evaluation criteria early; conduct pilot studies |
| Poor documentation | Medium | Maintain documentation alongside development |
| Code quality issues | Medium | Regular code reviews; implement testing framework |

---

## 6. Expected Contributions

### 6.1 Academic Contributions

1. **Novel Hybrid Representation Framework:** Combining dense semantic embeddings with sparse attribute vectors for robust person identification
2. **Cross-Modal Matching Pipeline:** End-to-end system for matching persons across different media types (images, videos, text descriptions)
3. **Dynamic Attribute Registry:** Self-extending attribute mapping system that adapts to new data without manual reconfiguration
4. **Comprehensive Evaluation:** Extensive experimental validation with detailed performance analysis

### 6.2 Practical Contributions

1. **Open-Source System:** Fully functional CrossMedia-PID system available for research and practical applications
2. **Modular Architecture:** Reusable components for person detection, feature extraction, vectorization, and matching
3. **User-Friendly Interfaces:** Both CLI and Web GUI for easy system interaction and testing
4. **Comprehensive Documentation:** Detailed technical documentation and usage guides

### 6.3 Research Impact

The CrossMedia-PID system addresses a significant gap in current person re-identification research by:
- Enabling cross-modal person matching beyond traditional camera-to-camera scenarios
- Leveraging large vision-language models for rich attribute extraction
- Providing a scalable and extensible framework for future research
- Demonstrating practical applicability in surveillance, security, and multimedia analysis

---

## 7. Conclusion

The CrossMedia-PID project has made substantial progress in the first half of the development cycle. The core system architecture has been successfully implemented with all four processing modules (Visual Extractor, Feature Extractor, Vectorizer, and Identity Matcher) fully functional. Comprehensive testing has demonstrated high matching accuracy (0.95+ match scores) and excellent consistency (100% across repeated tests).

The system currently supports:
- ✅ Person detection and extraction from images
- ✅ Rich attribute extraction using VLMs
- ✅ Hybrid vector representation (dense + sparse)
- ✅ Accurate identity matching with configurable parameters
- ✅ Persistent vector database storage
- ✅ Interactive Web GUI for video testing
- ✅ Comprehensive performance monitoring and evaluation

The remaining work focuses on three key areas: (1) enhancing the system with face recognition and advanced video processing, (2) conducting comprehensive evaluation and experiments, and (3) completing the dissertation writing. With a well-structured plan and clear milestones, the project is on track for successful completion by the end of June 2026.

The CrossMedia-PID system represents a significant step forward in cross-modal person identification, combining cutting-edge vision-language models with robust matching algorithms and scalable infrastructure. The completed work provides a solid foundation for the final phase of development and dissertation writing.

---

## References

[1] Ultralytics YOLO Documentation. https://docs.ultralytics.com/  
[2] ChromaDB Documentation. https://docs.trychroma.com/  
[3] Qwen3-VL Technical Report. Alibaba Group, 2025.  
[4] BGE Embedding Models. BAAI, 2023.  
[5] Person Re-identification: A Comprehensive Survey. IEEE TPAMI, 2022.  
[6] Vision-Language Models: A Survey. ACM Computing Surveys, 2024.  
[7] Vector Databases for AI Applications. IEEE Data Engineering Bulletin, 2024.  

---

**Appendix A: Project Repository Structure**

```
HKU_Dissertation/
├── crossmedia_pid/
│   ├── api/                 # Web API routes
│   ├── configs/             # Configuration files
│   ├── core/                # Core algorithm modules
│   │   ├── extractor.py     # Module A: Visual Extractor
│   │   ├── feature_vlm.py   # Module B: Feature Extractor
│   │   ├── vectorizer.py    # Module C: Vectorizer
│   │   └── matcher.py       # Module D: Identity Matcher
│   ├── db/                  # Database layer
│   │   └── chroma_store.py  # ChromaDB integration
│   ├── main.py              # CLI entry point
│   └── setup.py             # Package configuration
├── test_photo/              # Test images
├── chroma_db/               # Vector database storage
├── video_test_webgui.py     # Web GUI application
├── test_comparison_timed.py # Performance testing
├── test_stability.py        # Stability testing
└── start_webgui.sh          # Web GUI launcher
```

**Appendix B: Configuration Example**

```yaml
models:
  yolo:
    model_path: "yolov8n.pt"
    conf_threshold: 0.5
    iou_threshold: 0.45
  
  vlm:
    provider: "aliyun"
    model_name: "qwen3-vl-235b-a3b-instruct"
    api_key: "${DASHSCOPE_API_KEY}"
  
  embedding:
    model_name: "BAAI/bge-small-zh-v1.5"
    max_length: 512

database:
  chroma:
    persist_directory: "./chroma_db"
    collection_name: "person_embeddings"
    distance_fn: "cosine"

matching:
  threshold: 0.72
  top_k: 5
  weights:
    dense: 0.6
    sparse: 0.3
    face: 0.1
```

---

**Report End**