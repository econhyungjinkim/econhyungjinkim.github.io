---
layout: archive
title: "Scientific Software"
permalink: /code/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}






Python
======

R
======
  <ul>{% for R in site.posts %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>

Julia
======
  <ul>{% for post in site.julia %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>

Temp (Will be updated later)
======
  <ul>{% for post in site.publications %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
