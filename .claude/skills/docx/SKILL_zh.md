---
name: docx
description: "综合性的文档创建、编辑和分析工具，支持修订跟踪、评论、格式保留和文本提取。当Claude需要处理专业文档(.docx文件)时使用：(1) 创建新文档，(2) 修改或编辑内容，(3) 处理修订跟踪，(4) 添加评论，或任何其他文档任务"
license: 专有许可。完整条款请参阅LICENSE.txt
---

# DOCX 创建、编辑和分析

## 概述

用户可能要求你创建、编辑或分析 .docx 文件的内容。.docx 文件本质上是一个包含 XML 文件和其他资源的 ZIP 压缩包，你可以读取或编辑这些内容。针对不同的任务，你可以使用不同的工具和工作流程。

## 工作流程决策树

### 读取/分析内容
使用下面的"文本提取"或"原始XML访问"部分

### 创建新文档
使用"创建新的Word文档"工作流程

### 编辑现有文档
- **自己的文档 + 简单更改**
  使用"基础OOXML编辑"工作流程

- **他人的文档**
  使用**"修订工作流程"**（推荐默认选项）

- **法律、学术、商业或政府文档**
  使用**"修订工作流程"**（必需）

## 读取和分析内容

### 文本提取
如果你只需要读取文档的文本内容，应该使用 pandoc 将文档转换为 markdown 格式。Pandoc 提供了优秀的文档结构保留功能，并且可以显示修订跟踪：

```bash
# 将文档转换为带有修订跟踪的 markdown
pandoc --track-changes=all 文件路径.docx -o 输出.md
# 选项: --track-changes=accept/reject/all
```

### 原始XML访问
对于评论、复杂格式、文档结构、嵌入媒体和元数据，你需要原始XML访问。对于这些功能中的任何一种，你都需要解包文档并读取其原始XML内容。

#### 解包文件
`python ooxml/scripts/unpack.py <office文件> <输出目录>`

#### 关键文件结构
* `word/document.xml` - 主文档内容
* `word/comments.xml` - 在document.xml中引用的评论
* `word/media/` - 嵌入的图像和媒体文件
* 修订跟踪使用 `<w:ins>`（插入）和 `<w:del>`（删除）标签

## 创建新的Word文档

当从头开始创建新的Word文档时，使用**docx-js**，它允许你使用JavaScript/TypeScript创建Word文档。

### 工作流程
1. **必需 - 阅读整个文件**: 完整阅读[`docx-js.md`](docx-js.md)（约500行）。**在阅读此文件时绝不设置任何范围限制。** 在进行文档创建之前，阅读完整的文件内容以获取详细的语法、关键的格式规则和最佳实践。
2. 使用Document、Paragraph、TextRun组件创建JavaScript/TypeScript文件（你可以假设所有依赖项都已安装，但如果没有，请参阅下面的依赖项部分）
3. 使用Packer.toBuffer()导出为.docx

## 编辑现有的Word文档

当编辑现有的Word文档时，使用**Document库**（用于OOXML操作的Python库）。该库自动处理基础设施设置并提供文档操作方法。对于复杂场景，你可以通过库直接访问底层DOM。

### 工作流程
1. **必需 - 阅读整个文件**: 完整阅读[`ooxml.md`](ooxml.md)（约600行）。**在阅读此文件时绝不设置任何范围限制。** 阅读完整的文件内容以获取Document库API和直接编辑文档文件的XML模式。
2. 解包文档：`python ooxml/scripts/unpack.py <office文件> <输出目录>`
3. 使用Document库创建并运行Python脚本（参见ooxml.md中的"Document Library"部分）
4. 打包最终文档：`python ooxml/scripts/pack.py <输入目录> <office文件>`

Document库为常见操作提供了高级方法，并为复杂场景提供了直接的DOM访问。

## 文档审阅的修订工作流程

这个工作流程允许你在markdown中规划全面的修订跟踪，然后在OOXML中实施它们。**关键**：对于完整的修订跟踪，你必须系统地实施所有更改。

**批处理策略**：将相关更改分组为3-10个更改的批次。这使得调试可管理，同时保持效率。在移动到下一批之前测试每个批次。

**原则：最小化、精确的编辑**
当实施修订跟踪时，只标记实际更改的文本。重复未更改的文本使编辑更难审查且显得不专业。将替换分解为：[未更改的文本] + [删除] + [插入] + [未更改的文本]。通过从原始文本中提取`<w:r>`元素并重用它来保留未更改文本的原始运行RSID。

示例 - 在句子中将"30 days"更改为"60 days"：
```python
# 不好 - 替换整个句子
'<w:del><w:r><w:delText>The term is 30 days.</w:delText></w:r></w:del><w:ins><w:r><w:t>The term is 60 days.</w:t></w:r></w:ins>'

# 好 - 只标记更改的内容，为未更改的文本保留原始<w:r>
'<w:r w:rsidR="00AB12CD"><w:t>The term is </w:t></w:r><w:del><w:r><w:delText>30</w:delText></w:r></w:del><w:ins><w:r><w:t>60</w:t></w:r></w:ins><w:r w:rsidR="00AB12CD"><w:t> days.</w:t></w:r>'
```

### 修订跟踪工作流程

1. **获取markdown表示**：将文档转换为保留修订跟踪的markdown：
   ```bash
   pandoc --track-changes=all 文件路径.docx -o 当前.md
   ```

2. **识别和分组更改**：审查文档并识别所有需要的更改，将它们组织成逻辑批次：

   **定位方法**（用于在XML中查找更改）：
   - 章节/标题编号（例如，"第3.2节"、"第四条"）
   - 段落标识符（如果编号）
   - 使用独特周围文本的Grep模式
   - 文档结构（例如，"第一段"、"签名块"）
   - **不要使用markdown行号** - 它们不映射到XML结构

   **批次组织**（每组3-10个相关更改）：
   - 按章节："批次1：第2节修正"、"批次2：第5节更新"
   - 按类型："批次1：日期修正"、"批次2：当事方名称更改"
   - 按复杂性：从简单的文本替换开始，然后处理复杂的结构更改
   - 按顺序："批次1：第1-3页"、"批次2：第4-6页"

3. **阅读文档和解包**：
   - **必需 - 阅读整个文件**：完整阅读[`ooxml.md`](ooxml.md)（约600行）。**在阅读此文件时绝不设置任何范围限制。** 特别注意"Document Library"和"Tracked Change Patterns"部分。
   - **解包文档**：`python ooxml/scripts/unpack.py <文件.docx> <目录>`
   - **注意建议的RSID**：解包脚本将建议一个用于修订跟踪的RSID。复制此RSID以在步骤4b中使用。

4. **分批实施更改**：将更改逻辑分组（按章节、按类型或按位置）并在单个脚本中一起实施它们。这种方法：
   - 使调试更容易（更小的批次 = 更容易隔离错误）
   - 允许增量进度
   - 保持效率（3-10个更改的批处理大小效果很好）

   **建议的批次分组**：
   - 按文档章节（例如，"第3节更改"、"定义"、"终止条款"）
   - 按更改类型（例如，"日期更改"、"当事方名称更新"、"法律术语替换"）
   - 按位置（例如，"第1-3页的更改"、"文档上半部分的更改"）

   对于每批相关更改：

   **a. 将文本映射到XML**：在`word/document.xml`中grep文本以验证文本如何在`<w:r>`元素之间分割。

   **b. 创建并运行脚本**：使用`get_node`查找节点，实施更改，然后`doc.save()`。模式参见ooxml.md中的**"Document Library"**部分。

   **注意**：在编写脚本之前总是grep `word/document.xml`以获取当前行号并验证文本内容。每次脚本运行后行号都会改变。

5. **打包文档**：所有批次完成后，将解包的目录转换回.docx：
   ```bash
   python ooxml/scripts/pack.py 解包目录 审阅后文档.docx
   ```

6. **最终验证**：对完整文档进行全面检查：
   - 将最终文档转换为markdown：
     ```bash
     pandoc --track-changes=all 审阅后文档.docx -o 验证.md
     ```
   - 验证所有更改都已正确应用：
     ```bash
     grep "原始短语" 验证.md  # 应该找不到
     grep "替换短语" 验证.md  # 应该找到
     ```
   - 检查没有引入意外的更改

## 将文档转换为图像

要直观分析Word文档，使用两步过程将它们转换为图像：

1. **将DOCX转换为PDF**：
   ```bash
   soffice --headless --convert-to pdf 文档.docx
   ```

2. **将PDF页面转换为JPEG图像**：
   ```bash
   pdftoppm -jpeg -r 150 文档.pdf 页面
   ```
   这会创建像`页面-1.jpg`、`页面-2.jpg`等文件。

选项：
- `-r 150`：将分辨率设置为150 DPI（调整质量/大小平衡）
- `-jpeg`：输出JPEG格式（如果首选PNG，使用`-png`）
- `-f N`：要转换的第一页（例如，`-f 2`从第2页开始）
- `-l N`：要转换的最后一页（例如，`-l 5`在第5页停止）
- `页面`：输出文件的前缀

特定范围示例：
```bash
pdftoppm -jpeg -r 150 -f 2 -l 5 文档.pdf 页面  # 仅转换第2-5页
```

## 代码风格指南
**重要**：为DOCX操作生成代码时：
- 编写简洁的代码
- 避免冗长的变量名和冗余操作
- 避免不必要的打印语句

## 依赖项

必需的依赖项（如果不可用则安装）：

- **pandoc**：`sudo apt-get install pandoc`（用于文本提取）
- **docx**：`npm install -g docx`（用于创建新文档）
- **LibreOffice**：`sudo apt-get install libreoffice`（用于PDF转换）
- **Poppler**：`sudo apt-get install poppler-utils`（用于pdftoppm将PDF转换为图像）
- **defusedxml**：`pip install defusedxml`（用于安全XML解析）