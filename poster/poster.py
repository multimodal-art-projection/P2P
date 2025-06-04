import base64
import io
import json
import os
import re
import subprocess
import time
import cairosvg

from PIL import Image
from pdf2image import convert_from_path
from playwright.sync_api import sync_playwright
from pydantic import BaseModel, Field, create_model
from tqdm import tqdm

from langchain import hub

# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.prompts.image import ImagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain.output_parsers import OutputFixingParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


def create_dynamic_poster_model(sections: dict[str, str]) -> type[BaseModel]:
    """Dynamically create a Poster model based on sections returned by LLM."""
    fields = {
        "title": (str, Field(default="", description="Title of the paper")),
        "authors": (str, Field(default="", description="Authors of the paper")),
        "affiliation": (
            str,
            Field(default="", description="Affiliation of the authors"),
        ),
    }

    for section_name, description in sections.items():
        fields[section_name] = (str, Field(default="", description=description))

    return create_model("DynamicPoster", **fields)


def remove_think_tags(llm_output):
    if hasattr(llm_output, "content"):
        content = llm_output.content
        cleaned_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        cleaned_content = re.sub(r"<think>.*", "", cleaned_content, flags=re.DOTALL)
        return AIMessage(content=cleaned_content)
    elif isinstance(llm_output, str):
        cleaned_output = re.sub(r"<think>.*?</think>", "", llm_output, flags=re.DOTALL)
        cleaned_output = re.sub(r"<think>.*", "", cleaned_output, flags=re.DOTALL)
        return cleaned_output
    return llm_output


def replace_figures_in_markdown(
    markdown: str,
    figures: list[str],
) -> str:
    pattern = r"!\[(.*?)\]\((\d+)\)"

    def replacer(match):
        figure_index = int(match.group(2))
        if 0 <= figure_index < len(figures):
            return f"![{match.group(1)}]({figures[figure_index]})"
        return match.group(0)

    return re.sub(pattern, replacer, markdown)


def replace_figures_in_poster(
    poster: BaseModel,
    figures: list[str],
) -> BaseModel:
    for field in poster.model_fields:
        if hasattr(poster, field):
            value = getattr(poster, field)
            if isinstance(value, str):
                setattr(poster, field, replace_figures_in_markdown(value, figures))
    return poster


def replace_figures_size_in_markdown(
    markdown: str,
    figures: list[str],
) -> str:
    pattern = r"!\[(.*?)\]\((\d+)\)"

    def replacer(match):
        figure_index = int(match.group(2))
        if 0 <= figure_index < len(figures):
            data = base64.b64decode(figures[figure_index])
            image = Image.open(io.BytesIO(data))
            width, height = image.size
            return f"![{match.group(1)}, width = {width}, height = {height}, aspect ratio = {width / height:.4f}]({match.group(2)})"
        return match.group(0)

    return re.sub(pattern, replacer, markdown)


def replace_figures_size_in_poster(
    poster: BaseModel,
    figures: list[str],
) -> BaseModel:
    for field in poster.model_fields:
        if hasattr(poster, field):
            value = getattr(poster, field)
            if isinstance(value, str):
                setattr(poster, field, replace_figures_size_in_markdown(value, figures))
    return poster


def replace_figures_in_html(html: str, figures: list[str]) -> str:
    pattern = r"src=\"(\d+)\""

    def replacer(match):
        figure_index = int(match.group(1))
        if 0 <= figure_index < len(figures):
            return f'src="data:image/png;base64,{figures[figure_index]}"'
        return match.group(0)

    return re.sub(pattern, replacer, html)


def get_sizes(type: str, html: str) -> list[list[dict]]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.set_content(html)

        contents = page.query_selector_all(f".{type}-content")
        content_sizes = []

        for content in contents:
            groups = content.query_selector_all(f"> *")
            group_sizes = []

            for group in groups:
                is_group = group.evaluate(
                    f"element => element.classList.contains('{type}-group')"
                )
                if not is_group:
                    bounding_box = group.bounding_box()
                    group_sizes.append(
                        [
                            {
                                "width": bounding_box["width"],
                                "height": bounding_box["height"],
                            }
                        ]
                    )
                    continue

                group.evaluate("(element) => element.style.alignItems = 'start'")

                columns = group.query_selector_all(f".{type}-column")
                column_sizes = []

                for column in columns:
                    bounding_box = column.bounding_box()
                    column_sizes.append(
                        {
                            "width": bounding_box["width"],
                            "height": bounding_box["height"],
                        }
                    )

                group_sizes.append(column_sizes)

            content_sizes.append(group_sizes)

        browser.close()
        return content_sizes


def generate_html_v2(vendor: str, model: str, poster: BaseModel, figures: list[str]):
    if vendor == "openai":
        if "o1" in model or "o3" in model or "o4" in model:
            llm = ChatOpenAI(
                model=model,
                temperature=1,
                max_tokens=8000,
            )
        else:
            llm = BaseChatOpenAI(
                model=model,
                temperature=1,
                max_tokens=8000,
                # model_kwargs={
                #     "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
                # },
            )

    style = """<style>
      html {
        font-family: "Times New Roman", Times, serif;
        font-size: 16px;
      }

      body {
        width: 1280px;
        margin: 0;
      }

      ol,
      ul {
        margin-left: 0.5rem;
      }

      li {
        margin-bottom: 0.5rem;
      }

      img {
        width: calc(100% - 2rem);
        margin: 0.5rem 1rem;
      }

      .poster-header {
        padding: 2rem;
        text-align: center;
      }

      .poster-title {
        margin-bottom: 1rem;
        font-size: 1.875rem;
        font-weight: bold;
      }

      .poster-author {
        margin-bottom: 0.5rem;
      }

      .poster-content {
        padding: 1rem;
      }

      .section {
        margin-bottom: 1rem;
      }

      .section-title {
        padding: 0.5rem 1rem;
        font-weight: bold;
      }

      .section-content {
        margin: 0 1rem;
      }
    </style>
"""

    layout_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="You are a professional academic poster web page creator and your task is to generate the HTML code for a nicely laid out academic poster web page based on the object provided."
            ),
            HumanMessagePromptTemplate.from_template(
                """# Object Description
- The object contains several fields. Each field represents a section, except for the title, author and affiliation fields. The field name is the title of the section and the field value is the Markdown content of the section.
- The image in Markdown is given in the format ![alt_text, width = original_width, height = original_height, aspect ratio = aspect_ratio](image_index).

# HTML Structure
- Only generate the HTML code inside <body>, without any other things.
- Place title, author and affiliation inside <div class="poster-header">. Place title inside <div class="poster-title">, author inside <div class="poster-author"> and affiliation inside <div class="poster-affiliation">.
- Place content inside <div class="poster-content">.
- Place each section inside <div class="section">. Place section title inside <div class="section-title"> and section content inside <div class="section-content">.
- Use <p> for paragraphs.
- Use <ol> and <li> for ordered lists, and <ul> and <li> for unordered lists.
- Use <img src="image_index" alt="alt_text"> for images.
- Use <strong> for bold text and <em> for italic text.
- Do not use tags other than <div>, <p>, <ol>, <ul>, <li>, <img>, <strong>, <em>.
- Do not create any sections that are not in the object. Do not split or merge any existing sections.
- Sections and contents should be strictly equal to the object, and should be placed strictly in the order of the object.

# Color Specification
- Select at least 2 colors from the visual identity of the affiliation. If there are multiple affiliations, consider the most well-known one.
- For example, Tsinghua University uses #660874 and #d93379, Beihang University uses #005bac and #003da6, Zhejiang University uses #003f88 and #b01f24. These are just examples, you must pick colors from the actual visual identity of the affiliation.
- Add text and background color to poster header and section title using inline style. Use gradient to make the poster more beautiful.
- The text and background color of each section title should be the same.
- Do not add styles other than color, background, border, box-shadow.
- Do not add styles like width, height, padding, margin, font-size, font-weight, border-radius.

# Layout Specification
- Optionally, inside <div class="poster-content">, group sections into columns using <div class="poster-group" style="display: flex; gap: 1rem"> and <div class="poster-column" style="flex: 1">.
- You must determine the optimal number and flex grow value of columns to create a balanced poster layout. If one column becomes too tall, redistribute sections to other columns.
- There can be multiple groups with different number and flex grow of columns.
- Optionally, inside <div class="section-content">, group texts and images into columns using <div class="section-group" style="display: flex; gap: 0.5rem"> and <div class="section-column" style="flex: 1">.
- For example, if there are two images in two columns whose aspect ratios are 1.2 and 2 respectively, the flex grow of two columns should be 1.2 and 2 respectively, to make the columns have the same height.
- Calculate the size of each image based on column width and aspect ratios. Add comment <!-- width = display_width, height = display_height --> before each image.
- Rearrange the structure and order of sections, texts and images to make the height of each column in the same group approximately the same.
- For example, if there are too many images in one section that make the height of the column too large, group the images into columns.
- The display width of each image should not be too large or too small compared to its original width.
- DO NOT LEAVE MORE THAN 5% BLANK SPACE IN THE POSTER.
- Use a 3-column or 4-column layout with a landscape (horizontal) orientation for optimal visual presentation.

# Output Requirement
- Please output the result in the following format:
  <think>
    Think step by step, considering all structures and specifications listed above one by one.
    Calculate the width and height of each column, text and image in detail, based on given style.
  </think>
  ```html
    HTML code inside <body>.
  ```
- Please make the content in <think> as detailed and comprehensive as possible.

# Existing Style
{style}

# Object
{poster}
"""
            ),
        ]
    )
    layout_chain = layout_prompt | llm
    output = layout_chain.invoke({"style": style, "poster": poster}).content
    layout_prompt.append(
        MessagesPlaceholder(variable_name="react"),
    )

    HTML_TEMPLATE = """<!DOCTYPE html>
<html>
  <head>
    <title>Poster</title>
    {style}
    <script>
      MathJax = {{ tex: {{ inlineMath: [["$", "$"]] }} }};
    </script>
    <script
      id="MathJax-script"
      async
      src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
    ></script>
  </head>
  <body>
    {body}
  </body>
</html>
"""

    def get_content_sizes(sizes: list[list[dict]]) -> float:
        """Calculate the total content size from the sizes data structure"""
        return sum(
            column["width"] * column["height"]
            for content in sizes
            for group in content
            for column in group
        )

    def get_total_size(sizes: list[list[dict]]) -> float:
        """Calculate the total size including spacing from the sizes data structure"""
        return sum(
            (
                sum(column["width"] for column in group)
                * max((column["height"] for column in group), default=0)
            )
            for content in sizes
            for group in content
        )

    def calculate_blank_proportion(poster_sizes, section_sizes) -> float:
        """Calculate the proportion of blank space in the poster"""
        poster_content_sizes = get_content_sizes(poster_sizes)
        section_content_sizes = get_content_sizes(section_sizes)
        poster_total_size = get_total_size(poster_sizes)
        section_total_size = get_total_size(section_sizes)

        if poster_total_size == 0:
            return 1.0

        return (
            1.0
            - (poster_content_sizes - (section_total_size - section_content_sizes))
            / poster_total_size
        )

    max_attempts = 5
    attempt = 1

    while True:
        body = re.search(r"```html\n(.*?)\n```", output, re.DOTALL).group(1)

        html = HTML_TEMPLATE.format(style=style, body=body)
        html_with_figures = replace_figures_in_html(html, figures)

        poster_sizes = get_sizes("poster", html_with_figures)
        section_sizes = get_sizes("section", html_with_figures)

        proportion = calculate_blank_proportion(poster_sizes, section_sizes)
        if proportion < 0.15:
            print(
                f"Attempted {attempt} times, remaining {proportion:.0%} blank spaces."
            )
            return {"html": html, "html_with_figures": html_with_figures}

        attempt += 1
        if attempt > max_attempts:
            raise ValueError(f"Invalid blank spaces: {proportion:.0%}")

        react = [
            # AIMessage(""),
            HumanMessage(
                content=f"""# Previous Body
{body}

# Previous Size of Columns in Poster
{poster_sizes}

# Previous Size of Columns in Section
{section_sizes}

Now there are {proportion:.0%} blank spaces. Please regenerate the content to create a more balanced poster layout.
"""
            ),
        ]

        output = layout_chain.invoke(
            {"style": style, "poster": poster, "react": react}
        ).content


def take_screenshot(output: str, html: str):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 100})
        page.set_content(html)
        page.screenshot(
            type="png", path=output.replace(".json", ".png"), full_page=True
        )
        browser.close()


def replace_figures_in_svg(svg: str, figures: list[str]) -> str:
    pattern = r"href=\"(\d+)\""

    def replacer(match):
        figure_index = int(match.group(1))
        if 0 <= figure_index < len(figures):
            return f'href="data:image/png;base64,{figures[figure_index]}"'
        return match.group(0)

    return re.sub(pattern, replacer, svg)


def svg_to_png(output: str, svg: str):
    cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        write_to=output.replace(".json", ".png"),
        output_width=7000,
    )


def replace_figures_in_latex(latex: str, figures: list[str]) -> str:
    pattern = r"\\includegraphics(\[.*?\])?\{(\d+)\}"

    def replacer(match):
        figure_index = int(match.group(2))
        options = match.group(1) or ""
        if 0 <= figure_index < len(figures):
            return f"\\includegraphics{options}{{figure_{figure_index}.png}}"
        return match.group(0)

    return re.sub(pattern, replacer, latex)


def latex_to_png(output: str, latex: str):
    subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            f"-output-directory={os.path.dirname(output)}",
            output.replace(".json", ".tex"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    images = convert_from_path(output.replace(".json", ".pdf"), dpi=300)
    images[0].save(output.replace(".json", ".png"))


def generate_poster_v3(
    vendor: str,
    model: str,
    text_prompt: str,
    figures_prompt: str,
    pdf: str,
    figures: list[str],
    figures_index: list[str],
) -> dict:
    # Setup LLM
    if vendor == "openai":
        if "o1" in model or "o3" in model or "o4" in model:
            llm = ChatOpenAI(
                model=model,
                temperature=1,
                max_tokens=8000,
            )
        else:
            llm = BaseChatOpenAI(
                model=model,
                temperature=1,
                max_tokens=8000,
                # model_kwargs={
                #     "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
                # },
            )
    loader = PyMuPDFLoader(pdf)
    pages = loader.load()
    paper_content = "\n".join([page.page_content for page in pages])

    from .compress import compress_image

    figure_messages = [
        HumanMessagePromptTemplate(
            prompt=[
                ImagePromptTemplate(
                    input_variables=["figure"],
                    template={"url": "data:image/png;base64,{figure}"},
                ),
            ],
        ).format(figure=compress_image(figure, quality=85, max_size=(64, 64)))
        for figure in figures
    ]

    json_format_example = """
```json
{{
    "Introduction": "Brief overview of the paper's main topic and objectives.",
    "Methodology": "Description of the methods used in the research.",
    "Results": "Summary of the key findings and results."
}}
```
"""
    sections = None
    for _ in range(5):
        section_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content="You are an expert in academic paper analysis."),
                HumanMessagePromptTemplate.from_template(
                    """Please analyze the paper content and identify the key sections that should be included in the poster. 
For each section, provide a concise description of what should be included. First, determine the paper type:
- For methodology research papers: Focus on method description, experimental results, and research methodology.
- For benchmark papers: Highlight task definitions, dataset construction, and evaluation outcomes.
- For survey/review papers: Emphasize field significance, key developmental milestones, critical theories/techniques, current challenges, and emerging trends.

Note that the specific section names should be derived from the paper's content. Related sections can be combined to avoid fragmentation. Limit the total number of sections to maintain clarity. Do not include acknowledgements or references sections.

Return the result as a flat JSON object with section names as keys and descriptions as values, without nested structures. You MUST use Markdown code block syntax with the json language specifier.

Example format:
{json_format_example}

Paper content:
{paper_content}
"""
                ),
            ]
        )
        sections_response = llm.invoke(
            section_prompt.format(
                json_format_example=json_format_example, paper_content=paper_content
            )
        )

        json_pattern = r"```json(.*?)```"
        match = re.search(json_pattern, sections_response.content, re.DOTALL)
        if match:
            json_content = match.group(1)
        else:
            continue

        try:
            sections = eval(json_content.strip())
            if all(
                isinstance(k, str) and isinstance(v, str) for k, v in sections.items()
            ):
                break
        except Exception:
            continue

    if sections is None:
        raise ValueError("Failed to retrieve valid sections from LLM response.")

    DynamicPoster = create_dynamic_poster_model(sections)

    figures_description_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="You are an academic image analysis expert. Provide concise descriptions (under 100 words) of academic figures, diagrams, charts, or images. Identify what the figure displays, its likely purpose in academic literature, and highlight key data points or trends. Focus on clarity and academic relevance while maintaining precision in your analysis."
            ),
            HumanMessagePromptTemplate(
                prompt=[
                    # PromptTemplate(template="Describe this image:"),
                    ImagePromptTemplate(
                        input_variables=["image_data"],
                        template={"url": "data:image/png;base64,{image_data}"},
                    ),
                ],
            ),
        ]
    )

    use_claude = False
    mllm = BaseChatOpenAI(
        temperature=1,
        max_tokens=8000,
    )

    figures_with_descriptions = ""
    figure_list = []

    figures_description_cache = pdf.replace(".pdf", "_figures_description.json")
    if use_claude and os.path.exists(figures_description_cache):
        with open(figures_description_cache, "r") as f:
            figures_with_descriptions = f.read()
    else:
        figure_chain = figures_description_prompt | (mllm if use_claude else llm)
        for i, figure in enumerate(tqdm(figures, desc=f"处理图片 {pdf}")):
            figure_description_response = figure_chain.invoke({"image_data": figure})
            figures_with_descriptions += f"""
<figure_{i}>
{figure_description_response.content}
</figure_{i}>
"""
            figure_list.append(
                {"figure": figure, "description": figure_description_response.content}
            )
        if use_claude:
            with open(figures_description_cache, "w") as f:
                f.write(figures_with_descriptions)

    text_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="You are a helpful academic expert, who is specialized in generating a text-based paper poster, from given contents."
            ),
            HumanMessagePromptTemplate.from_template(
                """Below is the figures with descriptions in the paper:
<figures>
{figures}
</figures>

Below is the content of the paper:
<paper_content>
{paper_content}
</paper_content>

If figures can effectively convey the poster content, simplify the related text to avoid redundancy. Include essential mathematical formulas where they enhance understanding.

{format_instructions}

Ensure all sections are precise, concise, and presented in markdown format without headings."""
            ),
        ]
    )
    parser = PydanticOutputParser(pydantic_object=DynamicPoster)
    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
    text_prompt = text_prompt.partial(
        format_instructions=parser.get_format_instructions()
    )
    text_chain = text_prompt | llm | remove_think_tags | parser
    try:
        text_poster = text_chain.invoke(
            {"paper_content": paper_content, "figures": figures_with_descriptions}
        )
    except OutputParserException as e:
        text_poster = fixing_parser.parse(e.llm_output)

    figures_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "You are a helpful academic expert, who is specialized in generating a paper poster, from given contents and figures. "
            ),
            HumanMessagePromptTemplate.from_template(
                """Below is the figures with descriptions in the paper:
<figures>
{figures}
</figures>

I have already generated a text-based poster as follows:
<poster_content>
{poster_content}
</poster_content>

The paper content is as follows:
<paper_content>
{paper_content}
</paper_content>

Insert figures into the poster content using figure index notation as `![figure_description](figure_index)`. For example, `![Overview](0)`.
The figure_index MUST be an integer starting from 0, and no other text should be used in the figure_index position.
Each figure should be used at most once, with precise and accurate placement.
Prioritize pictures and tables based on their relevance and importance to the content.

{format_instructions}"""
            ),
        ]
    )
    figures_prompt = figures_prompt.partial(
        figures=figures_with_descriptions,
        format_instructions=parser.get_format_instructions(),
    )
    figures_chain = figures_prompt | llm | remove_think_tags | parser
    try:
        figures_poster = figures_chain.invoke(
            {"poster_content": text_poster, "paper_content": paper_content}
        )
    except OutputParserException as e:
        figures_poster = fixing_parser.parse(e.llm_output)

    return {
        "sections": sections,
        "figures": figure_list,
        "text_based_poster": text_poster,
        "image_based_poster": figures_poster,
    }
