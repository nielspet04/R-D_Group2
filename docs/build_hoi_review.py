from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
title = 'Hoi! Paper Review'
source = 'Tim Engelbracht et al. Hoi! - A Multimodal Dataset for Force-Grounded, Cross-View Articulated Manipulation. CVPR 2026, pp. 8880-8890.'
sections = [('Executive summary',
  ['Hoi! addresses a gap between human activity datasets, which show what people do, and robotics '
   'datasets, which record the physical information needed to carry out actions. Its main '
   'contribution is a dataset and capture system for articulated furniture manipulation. It '
   'contains 3,048 sequences involving 381 articulated objects in 38 environments. The same '
   'objects are manipulated using a bare hand, a hand with a wrist camera, a handheld UMI gripper, '
   'and the custom Hoi! gripper. Recordings cover egocentric and two exocentric views, with wrist '
   'views and additional sensing depending on the setup.',
   'The Hoi! gripper combines a six-axis force-torque sensor, two DIGIT tactile sensors, and wrist '
   'RGB-D sensing. Recordings are aligned in time and registered to high-resolution scene scans. '
   'Annotations include articulation type and axis, object-part masks, and language descriptions. '
   'The authors also open-source the gripper design, allowing researchers to reproduce or adapt '
   'the setup and collect additional data for specific objects or interaction conditions. '
   'Together, the dataset and capture system provide a basis for studying the link between visual '
   'observations, motion, and physical interaction across views and embodiments.',
   'The evaluations cover articulation estimation, tactile force estimation, and visual force '
   'estimation. Existing methods struggle to generalize to the interaction conditions in Hoi!, '
   'showing where current models need improvement. These results motivate the need for richer '
   'multimodal data and provide baselines for future work. The paper therefore offers a resource '
   'for developing and evaluating better models, although it does not yet show whether training on '
   'Hoi! improves robot control or transfer across embodiments.']),
 ('Strengths',
  ['The main strength is the combination of modalities and repeated interactions with the same '
   'objects across embodiments. Compared with video-only resources such as EgoExo4D or smaller '
   'articulation datasets such as RBO, Hoi! makes it possible to study visible actions together '
   'with contact and force information. Using the same objects also helps separate embodiment '
   'differences from differences in object identity. The portable gripper allows this data to be '
   'collected beyond a fixed laboratory setup, while the open-source design makes further targeted '
   'collection possible.',
   'The scene scans and sensor calibration are another strong point. They place camera '
   'observations, gripper trajectories, and force measurements in a common coordinate system. '
   'This makes it possible to relate measured forces to the object geometry and opening direction: '
   'for example, whether a force acts along a drawer\'s motion or pushes sideways. The scans also '
   'support annotation of the moving parts and articulation axes. The motion-capture evaluation '
   'reports positional RMSEs of 5-6 mm for head, wrist, and gripper trajectories (Table 3), '
   'providing evidence for tracking quality.',
   'The benchmarks give concrete examples of the research problems the dataset can support. '
   'Articulation estimation is affected by clutter, occlusion, and depth quality, while force '
   'prediction is challenged by unfamiliar contacts and loads. Reporting both raw and '
   'motion-projected forces also draws attention to the influence of the human operator on the '
   'measurements.']),
 ('Weaknesses and limitations',
  ['The dataset is designed to support transfer, but the experiments provide limited evidence of '
   'that transfer. Most recordings are human-operated, with only a small subset collected using a '
   'teleoperated Spot robot. A person holding a gripper does not have the same movement '
   'constraints as a robot. Force and tactile measurements are also only available for the Hoi! '
   'gripper condition, rather than for the bare-hand demonstrations. The recordings therefore link '
   'interactions at the object and task level, without providing identical force-labelled '
   'executions. A useful next step would be to train on one embodiment or view and test on '
   'another, followed by robot execution.',
   'Some benchmark comparisons are difficult to interpret because both the data and the input '
   "pipeline change. ArtiPoint's prismatic/revolute type recall drops from 68%/98% on Arti4D to "
   '26.9%/57.1% on Hoi! (Table 4). However, it uses scaled monocular depth on Hoi!, so this does '
   'not isolate the effect of more difficult scenes. Similarly, Sparsh is evaluated using forces '
   'combined across two opposing sensors and clipped to its expected range. As the authors '
   'acknowledge, this differs from its original benchmark. These results show generalization '
   'challenges, but do not establish how much training on Hoi! could reduce them.',
   'The choice of force target also needs further study. Projecting forces onto the motion '
   'direction may remove disturbances from the operator, but could also remove forces relevant to '
   "maintaining contact. ForceSight's RMSE decreases from 2.57 N on raw forces to 2.23 N on "
   'projected forces, compared with about 0.40 N on its original dataset (Table 6). ForceSight '
   'particularly struggles in locations with objects requiring higher operating forces, such as '
   'fridges, ovens, and magnetic drawers. This suggests limited generalization to more '
   'force-demanding mechanisms, although the evaluation does not isolate force magnitude from '
   'other differences between objects and scenes. Differences in '
   'force scale and target construction make this comparison difficult. Lower prediction error '
   'alone does not show that the target is more useful for manipulation; contact-level analysis '
   'and robot success rates would help assess this.',
   'Coverage remains selective despite the variety of objects. Collection mainly takes place in a '
   'furniture-store exhibition area and involves seven demonstrators. Repeated opening and closing '
   'actions give limited coverage of unusual mechanisms or recovery from failed interactions. '
   'Clear object-, environment-, and demonstrator-disjoint splits would help assess '
   'generalization. It would also be useful to examine the reported 10-25 ms synchronization '
   'accuracy around brief contact events, where timing errors could affect the force labels.']),
 ('Overall assessment',
  ['Hoi! is a useful dataset contribution because it brings visual, geometric, force, and tactile '
   'information together across several views and embodiments. The open-source gripper adds value '
   'by allowing the collection setup to be reused and extended. The evaluations show why better '
   'models and more varied interaction data are needed. Whether this resource leads to better '
   'manipulation and human-to-robot transfer is an important question for future work.']),
 ('Two anchor questions',
  ['1. What could a robot learn from watching a human open a drawer that it could not learn '
   'from the gripper recordings alone?',
   '2. Why do you think ForceSight struggles more with objects that require higher forces, '
   'and how could Hoi! help improve this?'])]

body = ParagraphStyle('Body', fontName='Times-Roman', fontSize=11, leading=13.7, spaceAfter=7)
heading = ParagraphStyle('Heading', fontName='Times-Bold', fontSize=12, leading=14, spaceBefore=8, spaceAfter=6, keepWithNext=True)
title_style = ParagraphStyle('Title', fontName='Times-Bold', fontSize=19, leading=22, spaceAfter=8)
ref_style = ParagraphStyle('Source', fontName='Times-Roman', fontSize=9, leading=11, spaceAfter=9)
story = [Paragraph(title, title_style), Paragraph(escape(source), ref_style)]
md = ['# '+title, '', source, '']
for i, (h, paragraphs) in enumerate(sections):
    if i == 2:
        story.append(PageBreak())
    story.append(Paragraph(h, heading))
    md += ['## '+h, '']
    for p in paragraphs:
        story.append(Paragraph(escape(p), body))
        md += [p, '']

def footer(c, doc):
    c.setFont('Times-Roman', 9)
    c.drawCentredString(A4[0]/2, 28, str(doc.page))

pdf_path = OUT/'Hoi_Paper_Review.pdf'
doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=49, leftMargin=49, topMargin=38, bottomMargin=43,
                        title='Hoi! Paper Review', author='')
doc.build(story, onFirstPage=footer, onLaterPages=footer)
(OUT/'Hoi_Paper_Review.md').write_text('\n'.join(md), encoding='utf-8')
reader = PdfReader(pdf_path)
print('Pages:', len(reader.pages))
for i, page in enumerate(reader.pages):
    print('Page', i+1, 'words:', len(page.extract_text().split()))
assert len(reader.pages) == 2, 'Review exceeds two pages'