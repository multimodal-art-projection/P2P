import gradio as gr
import tempfile
import os
import json
import shutil
from pathlib import Path
import base64
import logging
from main import generate_paper_poster


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_paper_to_poster(
    pdf_file, 
    model_choice, 
    figure_service_url,
    openai_api_key,
    openai_base_url
):
    """
    处理上传的PDF文件并生成海报 - 支持实时状态更新
    """
    if pdf_file is None:
        yield None, None, None, "❌ Please upload a PDF file first!"
        return
    
    if not openai_api_key.strip():
        yield None, None, None, "❌ Please enter your OpenAI API Key!"
        return
    
    if not figure_service_url.strip():
        yield None, None, None, "❌ Please enter the figure detection service URL!"
        return
    
    try:
        # 初始状态
        yield None, None, None, "🚀 Starting poster generation process..."
        
        # 配置OpenAI设置
        yield None, None, None, "⚙️ Configuring OpenAI API settings..."
        os.environ['OPENAI_API_KEY'] = openai_api_key.strip()
        if openai_base_url.strip():
            os.environ['OPENAI_BASE_URL'] = openai_base_url.strip()
        
        # 创建临时目录
        yield None, None, None, "📁 Creating temporary workspace..."
        temp_dir = tempfile.mkdtemp()
        
        # 保存上传的PDF文件
        yield None, None, None, "📄 Processing uploaded PDF file..."
        pdf_path = os.path.join(temp_dir, "paper.pdf")
        shutil.copy(pdf_file.name, pdf_path)
        
        # 开始调用生成函数
        yield None, None, None, "🔍 Extracting content from PDF and detecting figures..."
        
        # 调用原始生成函数
        poster, html = generate_paper_poster(
            url=figure_service_url,
            pdf=pdf_path,
            vendor="openai",
            model=model_choice,
            text_prompt="",  # 使用默认提示
            figures_prompt="",  # 使用默认提示
            output=""  # 不再使用
        )
        
        # 检查返回值是否为None
        if poster is None and html is None:
            yield None, None, None, "❌ Failed to generate poster! The paper processing returned no results. Please check:\n- PDF file format and content\n- Figure detection service availability\n- API key validity\n- Model configuration\n- Network connectivity"
            return
        
        # 图片处理完毕，开始生成JSON
        yield None, None, None, "🖼️ Image processing completed! Generating JSON structure..."
        
        # 将海报转换为JSON以便预览
        json_content = json.dumps(poster.model_dump(), indent=2, ensure_ascii=False)
        
        # JSON生成完毕，开始生成HTML
        yield None, None, None, "📋 JSON file generated successfully! Creating HTML poster..."
        
        # 创建持久化的临时文件用于下载
        json_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
        html_file = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8')
        
        # 写入内容到文件
        json_file.write(json_content)
        json_file.close()
        
        html_file.write(html)
        html_file.close()
        
        # 清理我们创建的临时目录
        yield None, None, None, "🧹 Cleaning up temporary files..."
        shutil.rmtree(temp_dir)
        
        # 最终完成
        yield (
            [json_file.name, html_file.name],
            json_content,
            html,
            "✅ Poster generated successfully! 🎉\n📥 Files are ready for download\n🎨 HTML preview is displayed below\n💡 Download the HTML file for best viewing experience"
        )
        
    except Exception as e:
        error_msg = f"❌ Error occurred during processing: {str(e)}"
        yield None, None, None, error_msg


# 创建Gradio界面
def create_interface():
    
    # JavaScript代码强制启用Light模式
    js_func = """
    function refresh() {
        const url = new URL(window.location);

        if (url.searchParams.get('__theme') !== 'light') {
            url.searchParams.set('__theme', 'light');
            window.location.href = url.href;
        }
    }
    """
    
    with gr.Blocks(
        title="P2P: Paper-to-Poster Generator", 
        theme=gr.themes.Default(),  # 使用Light主题
        js=js_func,  # 添加JavaScript强制Light模式
        css="""
        .gradio-container {
            max-width: 1600px !important;
        }
        .title {
            text-align: center;
            margin-bottom: 1rem;
        }
        .preview-container {
            min-height: 1000px;
            max-height: 1500px;
            overflow-y: auto;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            background-color: #fafafa;
        }
        .preview-container iframe {
            width: 100% !important;
            min-height: 1000px !important;
        }
        .config-section {
            margin-bottom: 2rem;
        }
        .status-updating {
            color: #2563eb;
            font-weight: 500;
        }
        """
    ) as demo:
        
        gr.HTML("""
        <div class="title">
            <h1>🎓 P2P: Paper-to-Poster Generator</h1>
            <p>Automatically convert academic papers into professional conference posters ✨</p>
            <p><a href="https://arxiv.org/abs/2505.17104" target="_blank">📄 View Research Paper</a></p>
        </div>
        """)
        
        # 配置区域 - 水平布局
        with gr.Row(elem_classes=["config-section"]):
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Input Configuration")
                
                # 文件上传
                pdf_input = gr.File(
                    label="Upload PDF Paper File",
                    file_types=[".pdf"],
                    file_count="single"
                )
                
                # OpenAI API配置
                gr.Markdown("#### 🔑 OpenAI API Configuration")
                openai_api_key = gr.Textbox(
                    label="OpenAI API Key",
                    placeholder="sk-...",
                    type="password",
                    info="Enter your OpenAI API key"
                )
                
                openai_base_url = gr.Textbox(
                    label="OpenAI Base URL (Optional)",
                    placeholder="https://api.openai.com/v1",
                    value="https://api.openai.com/v1",
                    info="Modify this URL if using proxy or other OpenAI-compatible services"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Model Configuration")
                
                # 模型选择
                model_choice = gr.Textbox(
                    label="AI Model Name",
                    value="gpt-4o-mini",
                    placeholder="e.g., gpt-4o-mini, gpt-4o, gpt-3.5-turbo, claude-3-sonnet",
                    info="Enter the AI model name you want to use"
                )
                
                # 图片检测服务URL
                figure_url = gr.Textbox(
                    label="Figure Detection Service URL",
                    placeholder="Enter the URL of figure detection service",
                    info="Used to extract images and tables from PDF"
                )
                
                # 生成按钮
                generate_btn = gr.Button(
                    "🚀 Generate Poster", 
                    variant="primary", 
                    size="lg"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Results & Downloads")
                
                # 状态消息
                status_msg = gr.Textbox(
                    label="Status Information",
                    interactive=False,
                    lines=4,
                    show_copy_button=True
                )
                
                # 文件下载
                output_files = gr.File(
                    label="📥 Download Generated Files (JSON & HTML)",
                    file_count="multiple",
                    interactive=False,
                    show_label=True
                )
                
                # JSON预览 - 压缩到侧边栏
                with gr.Accordion("📋 JSON Structure", open=False):
                    json_preview = gr.Code(
                        label="",
                        language="json",
                        lines=10,
                        show_label=False
                    )
        
        # 预览区域 - 跨栏全宽显示
        gr.Markdown("### 🎨 HTML Poster Preview")
        
        gr.Markdown("**💡 Recommended: Download the HTML file from above and open it in your browser for optimal viewing experience**")
        
        # HTML预览 - 全宽显示
        html_preview = gr.HTML(
            label="",
            show_label=False,
            elem_classes=["preview-container"]
        )
        
        # 使用说明
        gr.Markdown("""
        ### 📖 Usage Instructions
        
        1. **Upload PDF File**: Select the academic paper PDF you want to convert
        2. **Configure OpenAI API**: Enter your API Key and Base URL (if needed)
        3. **Select Model**: Enter model name manually, such as gpt-4o-mini, gpt-4o, claude-3-sonnet, etc.
        4. **Set Figure Service**: Enter the URL of the figure detection service
        5. **Generate Poster**: Click the generate button and wait for processing
        6. **Download Results**: Download the generated JSON and HTML files from the download section
        7. **Full Preview**: Download the HTML file and open it in your browser for the best viewing experience
        
        ⚠️ **Important Notes**:
        - Generated Poster is recommended to be viewed in fullscreen mode or download the HTML file to view in browser
        - Requires a valid OpenAI API key with sufficient balance
        - Figure detection service URL is required for extracting images from PDFs
        - Processing time depends on paper length and complexity (usually 3-6 minutes)
        - Ensure the model name is correct and supported by your API
        - Download the HTML file and open it in your browser for the best viewing experience
        
        💡 **Tips**: 
        - Recommended to use gpt-4o-mini model for cost-effective testing
        - Recommended to use Claude model for better performance
        - Modify Base URL if using domestic proxy services
        - Supports any OpenAI-compatible model names
        - Can use Claude, Gemini and other models (requires corresponding API configuration)
        - The HTML preview below shows how your poster will look with maximum width for better viewing
        - Download the HTML file from the "Download Generated Files" section for standalone viewing
        """)
        
        # 绑定事件
        generate_btn.click(
            fn=process_paper_to_poster,
            inputs=[
                pdf_input, 
                model_choice, 
                figure_url,
                openai_api_key,
                openai_base_url
            ],
            outputs=[
                output_files,
                json_preview, 
                html_preview,
                status_msg
            ]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )