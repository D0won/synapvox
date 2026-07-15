import ReactMarkdown from 'react-markdown'
import rehypeKatex from 'rehype-katex'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import 'katex/dist/katex.min.css'
import './MarkdownContent.css'

function normalizeMathDelimiters(markdown: string): string {
  // remark-math uses $...$/$$...$$. OpenAI may still return the equivalent
  // LaTeX \(...\)/\[...\] delimiters, so normalize only outside code spans.
  return markdown
    .split(/(```[\s\S]*?```|`[^`\n]*`)/g)
    .map((part, index) => {
      if (index % 2 === 1) return part
      return part
        .replace(/\\\[([\s\S]*?)\\\]/g, (_, math: string) => `\n$$${math.trim()}$$\n`)
        .replace(/\\\(([\s\S]*?)\\\)/g, (_, math: string) => `$${math.trim()}$`)
    })
    .join('')
}

export function MarkdownContent({
  children,
  className = '',
}: {
  children: string
  className?: string
}): React.JSX.Element {
  return (
    <div className={`markdown-content ${className}`.trim()}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          a: ({ children: linkText, ...props }) => (
            <a {...props} target="_blank" rel="noreferrer">
              {linkText}
            </a>
          ),
        }}
      >
        {normalizeMathDelimiters(children)}
      </ReactMarkdown>
    </div>
  )
}
