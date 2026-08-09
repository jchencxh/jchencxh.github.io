---
permalink: /
author_profile: true
hide_author_avatar: true
redirect_from: 
  - /about/
  - /about.html
---

{% assign latest_post = site.posts | first %}
{% if latest_post %}
## Latest Blog Post

{% assign post = latest_post %}
{% include archive-single.html %}
{% endif %}
