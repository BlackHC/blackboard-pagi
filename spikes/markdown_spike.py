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

import markdown
from markdown.extensions.wikilinks import WikiLinkExtension

#%%

markdown_example_with_nested_sections_and_wikilinks = """
# Section 1

Text A

Text A.1

Text A.2

## Section 1.1

Text B

### Section 1.1.1

Text C

#### Section

Text D

## Section 1.2

Text E

# Section 2

Text F

## Section 2.1

Text G

[[test]]

[[hello]]
"""

html_example_with_nested_sections_and_wikilinks = markdown.markdown(
    markdown_example_with_nested_sections_and_wikilinks,
    extensions=[WikiLinkExtension()],
)

print(html_example_with_nested_sections_and_wikilinks)

#%%

# Parse html_example_with_nested_sections_and_wikilinks
import bs4

parsed_html = bs4.BeautifulSoup(html_example_with_nested_sections_and_wikilinks, "html.parser")

# Get all sections and the text following (until the next section)
sections = parsed_html.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
sections_and_text = []
for section in sections:
    # Get the text following the section
    text = ""
    print(section.next_siblings)
    for sibling in section.next_siblings:
        if sibling.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            break
        text += sibling.text
    sections_and_text.append((section, text))

sections_and_text
