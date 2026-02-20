from sqlalchemy import ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Team(Base):
    __tablename__ = "teams"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    name: Mapped[str] = mapped_column(String(100))
    city: Mapped[str] = mapped_column(String(100))
    abbreviation: Mapped[str] = mapped_column(String(10))
    players: Mapped[list["Player"]] = relationship(back_populates="current_team")


class Player(Base):
    __tablename__ = "players"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    first_name: Mapped[str] = mapped_column(String(100))
    last_name: Mapped[str] = mapped_column(String(100))
    height: Mapped[str | None] = mapped_column(String(10), nullable=True)
    weight: Mapped[str | None] = mapped_column(String(10), nullable=True)
    birthdate: Mapped[str | None] = mapped_column(String(20), nullable=True)
    current_team_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("teams.id"), nullable=True
    )
    current_team: Mapped["Team | None"] = relationship(back_populates="players")
    box_scores: Mapped[list["BoxScore"]] = relationship(back_populates="player")


class Game(Base):
    __tablename__ = "games"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    home_team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id"))
    away_team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id"))
    date: Mapped[str] = mapped_column(String(20))
    season: Mapped[str] = mapped_column(String(10))
    playoff_round: Mapped[str | None] = mapped_column(String(50), nullable=True)
    home_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_team: Mapped["Team"] = relationship(foreign_keys=[home_team_id])
    away_team: Mapped["Team"] = relationship(foreign_keys=[away_team_id])
    box_scores: Mapped[list["BoxScore"]] = relationship(back_populates="game")


class BoxScore(Base):
    __tablename__ = "box_scores"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(Integer, ForeignKey("players.id"))
    game_id: Mapped[int] = mapped_column(Integer, ForeignKey("games.id"))
    team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id"))
    points: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rebounds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    assists: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fgm: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fga: Mapped[int | None] = mapped_column(Integer, nullable=True)
    three_pm: Mapped[int | None] = mapped_column(Integer, nullable=True)
    three_pa: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ftm: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fta: Mapped[int | None] = mapped_column(Integer, nullable=True)
    steals: Mapped[int | None] = mapped_column(Integer, nullable=True)
    blocks: Mapped[int | None] = mapped_column(Integer, nullable=True)
    turnovers: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pf: Mapped[int | None] = mapped_column(Integer, nullable=True)
    plus_minus: Mapped[int | None] = mapped_column(Integer, nullable=True)
    minutes: Mapped[str | None] = mapped_column(String(10), nullable=True)
    starter: Mapped[bool | None] = mapped_column(nullable=True)
    player: Mapped["Player"] = relationship(back_populates="box_scores")
    game: Mapped["Game"] = relationship(back_populates="box_scores")
    team: Mapped["Team"] = relationship()
    __table_args__ = (UniqueConstraint("player_id", "game_id"),)
