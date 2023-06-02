#  Blackboard-PAGI - LLM Proto-AGI using the Blackboard Pattern
#  Copyright (c) 2023. Andreas Kirsch
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import pynecone as pc
from pydantic import BaseModel, Field


class FlameGraphNode(BaseModel):
    name: str
    value: int
    children: list["FlameGraphNode"]
    color: str | None = None
    backgroundColor: str | None = Field(None, alias="background_color")
    tooltip: str | None = None
    id: str | None = None


class FlameGraph(pc.Component):
    library = "react-flame-graph"
    tag = "FlameGraph"

    data: pc.Var[dict]
    height: pc.Var[int]
    width: pc.Var[int]

    @classmethod
    def get_controlled_triggers(cls) -> dict[str, pc.Var]:
        return {"on_change": pc.EVENT_ARG}


flame_graph = FlameGraph.create
