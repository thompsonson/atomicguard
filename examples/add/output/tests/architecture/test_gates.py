"""
Test suite to enforce architecture gates defined in the project.
"""

import pytest
from pytestarch import Rule, get_evaluable_architecture


@pytest.fixture(scope="module")
def evaluable():
    return get_evaluable_architecture("/project", "/project/src")


def test_gate1_domain_no_infra_imports(evaluable):
    rule = (
        Rule()
        .modules_that()
        .are_sub_modules_of("domain")
        .should_not()
        .import_modules_that()
        .are_sub_modules_of("infrastructure")
    )
    rule.assert_applies(evaluable)


def test_gate2_application_no_direct_infra_imports(evaluable):
    rule = (
        Rule()
        .modules_that()
        .are_sub_modules_of("application")
        .should_not()
        .import_modules_that()
        .are_sub_modules_of("infrastructure")
    )
    rule.assert_applies(evaluable)


def test_gate3_dependency_direction(evaluable):
    rule = Rule().modules_that().are_sub_modules_of(
        "infrastructure"
    ).should_not().import_modules_that().are_sub_modules_of(
        "application"
    ) and Rule().or_().modules_that().are_sub_modules_of(
        "application"
    ).should_not().import_modules_that().are_sub_modules_of("domain")
    rule.assert_applies(evaluable)


def test_gate4_entity_containment(evaluable):
    rule = (
        Rule()
        .classes_that()
        .have_name_matching(".*Entity")
        .should_be_in_packages(["domain.entities"])
    )
    rule.assert_applies(evaluable)


def test_gate5_repository_containment(evaluable):
    rule = (
        Rule()
        .classes_that()
        .have_name_matching(".*Repository")
        .should_be_in_packages(["domain.interfaces"])
    ) + (
        Rule()
        .or_()
        .modules_that()
        .are_sub_modules_of("infrastructure.persistence")
        .should_import_classes_from_packages(["domain.interfaces"])
    )
    rule.assert_applies(evaluable)


def test_gate6_value_object_immutability(evaluable):
    rule = (
        Rule()
        .classes_that()
        .are_sub_classes_of("dataclasses.dataclass")
        .should_be_decorated_with("@dataclass(frozen=True)")
        .and_()
        .should_be_in_packages(["domain.value_objects"])
    )
    rule.assert_applies(evaluable)


def test_gate7_use_case_naming(evaluable):
    rule = (
        Rule()
        .classes_that()
        .have_name_matching(".*UseCase|.*Handler")
        .should_be_in_packages(["application"])
    )
    rule.assert_applies(evaluable)


def test_gate8_database_access_in_infrastructure(evaluable):
    rule = (
        Rule()
        .modules_that()
        .import_modules_matching("sqlalchemy|pymongo")
        .should_be_sub_modules_of("infrastructure.persistence")
    )
    rule.assert_applies(evaluable)
