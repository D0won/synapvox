// CitationDrawer — AI 답변의 인용 칩을 눌렀을 때, 노드 클릭 시의 DetailDrawer와
// 동일한 종이 드로어 인터페이스로 근거(fact) 내용을 보여준다. 마크업·클래스를
// DetailDrawer와 같은 detail.css 토큰으로 맞춰 두 화면이 한 시스템으로 보이게 한다.
// 헤더 ✕와 Esc 모두 닫기.
import { useEffect } from 'react'
import type { JSX } from 'react'
import './detail.css'

export function CitationDrawer(props: {
  n: number
  title: string
  fact: string
  onClose(): void
}): JSX.Element {
  const { n, title, fact, onClose } = props

  // Esc closes while the drawer is open (mirrors the header ✕).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <div className="detail" role="complementary" aria-label="출처">
      <header className="detail__head">
        <span className="detail__kind">출처 [{n}]</span>
        <button type="button" className="detail__close" onClick={onClose} aria-label="닫기">
          ✕
        </button>
      </header>

      <div className="detail__body">
        <h2 className="detail__title">{title}</h2>

        {/* 개념 화면(ConceptView)과 동일하게 제목 아래 detail__summary로 본문을 보여준다. */}
        {fact ? (
          <p className="detail__summary">{fact}</p>
        ) : (
          <p className="detail__summary detail__summary--empty">
            이 답변은 근거 내용이 저장되기 전에 생성되어 본문을 표시할 수 없습니다.
          </p>
        )}
      </div>
    </div>
  )
}
