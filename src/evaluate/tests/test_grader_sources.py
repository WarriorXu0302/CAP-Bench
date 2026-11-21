"""Offline regressions for source alternatives and URL formats in issue #5."""
import unittest

from grader_test_utils import grade, task


class SourceRubrics(unittest.TestCase):
    def test_restaurant_sources_are_alternatives(self):
        module = task('bc9c8a')
        yelp = 'https://www.yelp.com/biz/diner'
        tripadvisor = 'https://www.tripadvisor.com/Restaurant_Review-d123-Diner.html'
        for links in ((yelp,), (tripadvisor,), (yelp, tripadvisor)):
            with self.subTest(links=links):
                rows = [dict(name='Diner', rating='4.5', distance_from_station='2 mi', link=url)
                        for url in links]
                result = grade(module, ' '.join(links), route_and_stations={'restaurants': rows})
                self.assertEqual(result.find_node('restaurant_search').score, 1)
                self.assertEqual(result.find_node('restaurant_search_filters').score, 1)
                self.assertEqual(result.find_node('restaurant_output_validation').score, 1)
                self.assertEqual(len(result.root.children), 6)

        row = dict(name='Diner', link='https://example.org/yelp', review_source='Yelp')
        result = grade(module, row['link'], route_and_stations={'restaurants': [row]})
        self.assertEqual(result.find_node('restaurant_search_source').score, 0)
        result = grade(module, 'Ratings from Yelp', route_and_stations={'restaurants': [row]})
        self.assertEqual(result.find_node('restaurant_search_source').score, 1)
        self.assertEqual(result.find_node('restaurant_search_filters').score, 0)

    def test_mayo_citations_without_site_name_in_prose(self):
        module = task('978ccb')
        for domain, expected in [('www.mayoclinic.org', 1), ('mayoclinic.org.example.com', 0)]:
            with self.subTest(domain=domain):
                links = {name + '_link': f'https://{domain}/drugs-supplements/{name}'
                         for name in ('warfarin', 'simvastatin')}
                result = grade(module, ' '.join(links.values()), mayo_guidelines=links)
                self.assertEqual(result.find_node('mayo_action_navigate_drugs').score, expected)
                self.assertEqual(result.find_node('mayo_action_find_drugs').score, expected)

    def test_metacritic_pc(self):
        check = task('1a7a24').looks_like_metacritic_pc_url
        for url in ('https://metacritic.com/game/pc/balatro/',
                    'https://www.metacritic.com/game/balatro/?sort=date&platform=pc'):
            self.assertTrue(check(url), url)
        for url in ('https://metacritic.com/game/balatro/?platform=ps5',
                    'https://example.org/metacritic.com/game/pc/balatro',
                    'https://metacritic.com/movie/pc/title/', 'https://[invalid'):
            self.assertFalse(check(url), url)

    def test_twitch_filter_variants(self):
        check = task('1a7a24').has_twitch_all_time_filter
        for evidence in ('All Time', 'all-time', 'all_time', 'All–Time', 'alltime', 'range=all',
                         'https://www.twitch.tv/directory/category/balatro/clips?range=all'):
            self.assertTrue(check(evidence), evidence)
        for evidence in ('All timers', 'smalltime', 'https://twitch.tv/clips?range=7d',
                         'https://example.org/clips?range=all', 'range=all_months'):
            self.assertFalse(check(evidence), evidence)

    def test_google_flights_urls(self):
        check = task('f028d5').is_google_flights_url
        for url in ('https://www.google.com/flights', 'https://www.google.com/travel/flights/search?tfs=foo',
                    'https://flights.google.com/'):
            self.assertTrue(check(url), url)
        for url in ('https://google.com/maps', 'https://google.com/travel/flights-other',
                    'https://example.org/?next=https://google.com/flights', 'https://google.com.example.com/flights'):
            self.assertFalse(check(url), url)

    def test_twitch_filter_evidence_applies_to_its_game(self):
        module = task('1a7a24')
        twitch_listing = 'https://twitch.tv/directory/category/balatro/clips?range=all'
        unrelated = 'https://example.com/animal-well?range=all'
        answer = f'Balatro clips: {twitch_listing}. Animal Well: {unrelated}.'
        materials = dict(youtube_link='https://youtu.be/abc', youtube_channel='Official',
                         youtube_trailer_title='Launch trailer', metacritic_user_score='8',
                         metacritic_review_summary='A ten-point review',
                         metacritic_link='https://metacritic.com/game/balatro/?platform=pc')
        data = {
            'balatro': dict(materials, twitch_time_filter_evidence=twitch_listing),
            'animal_well': dict(materials, twitch_time_filter_evidence='range=all'),
        }
        result = grade(module, answer, all_games_materials=data)
        self.assertEqual(result.find_node('balatro_twitch_action_alltime_filter').score, 1)
        self.assertEqual(result.find_node('animal_well_twitch_action_alltime_filter').score, 0)

    def test_zillow_listing_urls(self):
        check = task('e71ffc').is_zillow_rental_url
        for path in ('/b/building', '/rentals/one', '/homedetails/address/123_zpid/'):
            self.assertTrue(check('https://www.zillow.com' + path), path)
        for url in ('https://zillow.com/homedetails/', 'https://zillow.com/phoenix-az/rentals/',
                    'https://example.org/b/house?source=zillow.com', None):
            self.assertFalse(check(url), url)

    def test_zillow_homedetails_still_require_rents_and_links(self):
        module = task('e71ffc')
        rows = [dict(address=f'{n} Main St, Phoenix AZ', parking_description='RV gate',
                     monthly_rent='$3,000', zillow_link=f'https://zillow.com/homedetails/{n}_zpid/')
                for n in (111, 222, 333)]
        answer = 'Zillow Phoenix AZ, facts and features, RV parking.'
        result = grade(module, answer, zillow_properties={'properties': rows})
        self.assertEqual(result.find_node('zillow_section').score, 1)
        rows[-1]['zillow_link'] = None
        result = grade(module, answer, zillow_properties={'properties': rows})
        self.assertEqual(result.find_node('zillow_links_valid').score, 0)
        rows[-1]['monthly_rent'] = None
        result = grade(module, answer, zillow_properties={'properties': rows})
        self.assertEqual(result.find_node('zillow_budget_range').score, 0)
