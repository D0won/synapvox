"""시간별·맥락별 조회 쿼리 (graphrag Baseline: 시간별/맥락별 정리의 축).

- timeline: FOLLOWS 축 (시간별 정리)
- meetings_by_topic: DISCUSSES 축 (맥락별 정리)
- decision_history: SUPERSEDES 체인 (결정 번복 이력)
"""

from . import schema as S


def timeline(driver, project_id: str) -> list[dict]:
    with driver.session() as s:
        rows = s.run(
            f"MATCH (m:{S.MEETING} {{project_id:$pid}}) "
            f"OPTIONAL MATCH (m)-[:{S.DISCUSSES}]->(t:{S.TOPIC}) "
            f"RETURN m.meeting_id AS meeting_id, m.date AS date, m.title AS title, "
            f"       m.summary AS summary, collect(DISTINCT t.name) AS topics "
            f"ORDER BY m.date, m.meeting_id", pid=project_id)
        return [r.data() for r in rows]


def meetings_by_topic(driver, project_id: str, topic_name: str) -> list[dict]:
    with driver.session() as s:
        rows = s.run(
            f"MATCH (m:{S.MEETING} {{project_id:$pid}})-[:{S.DISCUSSES}]->(t:{S.TOPIC}) "
            f"WHERE t.name=$name OR $name IN coalesce(t.aliases, []) "
            f"RETURN DISTINCT m.meeting_id AS meeting_id, m.date AS date, m.title AS title "
            f"ORDER BY m.date", pid=project_id, name=topic_name)
        return [r.data() for r in rows]


def decision_history(driver, project_id: str) -> list[dict]:
    with driver.session() as s:
        rows = s.run(
            f"MATCH (d:{S.DECISION} {{project_id:$pid}}) "
            f"OPTIONAL MATCH (d)-[:{S.SUPERSEDES}]->(prev:{S.DECISION}) "
            f"OPTIONAL MATCH (d)-[:{S.DECIDED_IN}]->(m:{S.MEETING}) "
            f"RETURN d.decision_id AS decision_id, d.statement AS statement, d.date AS date, "
            f"       prev.decision_id AS supersedes, m.meeting_id AS meeting_id "
            f"ORDER BY d.date", pid=project_id)
        return [r.data() for r in rows]
